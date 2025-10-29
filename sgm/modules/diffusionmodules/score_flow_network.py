"""
Score-based Network using Normalizing Flow
Replaces UNet while maintaining compatible interface with official DiffusionEngine
"""

import torch
import torch.nn as nn
from typing import Optional

from ...modules.flows import NormalizingFlow
from ...modules.flows.flow_layers import get_activation


class ScoreFlowNetwork(nn.Module):
    """
    Score network based on Normalizing Flow
    Compatible interface with UNet for official DiffusionEngine

    Key features:
    - Exact gradient computation via autograd
    - Energy-based score function
    - Compatible with official denoiser, loss, sampler
    """

    def __init__(
        self,
        in_channels: int,
        model_channels: int = 256,
        n_flows: int = 3,
        activation: str = 'softplus',
        activation_params: Optional[dict] = None,
        num_classes: Optional[int] = None,
        use_spatial_transformer: bool = False,
        transformer_depth: int = 1,
        context_dim: Optional[int] = None,
        sigma_data: float = 0.5,  # For EDM scaling
        **kwargs,  # Accept unused UNet params for compatibility
    ):
        """
        Args:
            in_channels: Input channels (e.g., 4 for latent, 3 for RGB)
            model_channels: Base hidden dimension
            n_flows: Number of flow layers
            activation: Activation function name
            activation_params: Parameters for activation
            num_classes: Number of classes for class conditioning
            use_spatial_transformer: Whether to use cross-attention (compatibility)
            transformer_depth: Transformer depth (compatibility)
            context_dim: Context dimension for cross-attention
        """
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_classes = num_classes
        self.use_spatial_transformer = use_spatial_transformer
        self.context_dim = context_dim
        self.sigma_data = sigma_data

        # Timestep (sigma) embedder
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Class embedder (if needed)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # Context projector (for cross-attention conditioning)
        if use_spatial_transformer and context_dim is not None:
            self.context_proj = nn.Linear(context_dim, model_channels)
        else:
            self.context_proj = None

        # Calculate flow input dimension
        # We'll flatten spatial dimensions and process as 1D
        # flow_input = flattened_data + time_emb + (optional) context
        flow_input_dim = in_channels + time_embed_dim
        if self.context_proj is not None:
            flow_input_dim += model_channels

        # Create activation function
        if activation_params is None:
            activation_params = {}
        activation_fn = get_activation(activation, **activation_params)

        # Normalizing Flow
        self.flow = NormalizingFlow(
            dim=flow_input_dim,
            n_flows=n_flows,
            activation=activation_fn,
        )

        # Score projector: enhance exact gradient with MLP
        self.score_projector = nn.Sequential(
            nn.Linear(flow_input_dim, model_channels * 2),
            nn.SiLU(),
            nn.Linear(model_channels * 2, in_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass compatible with UNet interface

        Args:
            x: Input tensor [B, C, H, W] (noised latent from denoiser)
            timesteps: Timestep/sigma values [B] (c_noise from denoiser scaling)
            context: Cross-attention context [B, L, D] (text embeddings)
            y: Class labels [B] or class embeddings [B, D]

        Returns:
            denoised: Denoised output [B, C, H, W]
        """
        batch_size = x.shape[0]
        original_shape = x.shape
        spatial_size = x.shape[2] * x.shape[3] if len(x.shape) > 2 else 1

        # Flatten spatial dimensions: [B, C, H, W] -> [B, C*H*W]
        if len(x.shape) > 2:
            x_flat = x.reshape(batch_size, -1)
        else:
            x_flat = x

        # Enable gradient tracking for exact score computation
        x_flat = x_flat.requires_grad_(True)

        # Embed timesteps
        if timesteps is not None:
            # timesteps might be already embedded by denoiser (c_noise)
            if timesteps.dim() == 1 or (timesteps.dim() == 2 and timesteps.shape[-1] == 1):
                # Need to embed: [B] or [B, 1]
                if timesteps.dim() == 1:
                    timesteps = timesteps.unsqueeze(-1)

                # Sinusoidal position embedding
                half_dim = self.model_channels // 2
                emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
                emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
                emb = timesteps.float() * emb.unsqueeze(0)
                emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

                if self.model_channels % 2 == 1:  # zero pad if odd
                    emb = torch.nn.functional.pad(emb, (0, 1))

                time_emb = self.time_embed(emb)  # [B, time_embed_dim]
            else:
                # Already high-dimensional, might be pre-embedded
                time_emb = timesteps
                if time_emb.shape[-1] != self.model_channels * 4:
                    # Need to project
                    time_emb = self.time_embed(time_emb)
        else:
            time_emb = torch.zeros(batch_size, self.model_channels * 4, device=x.device)

        # Add class conditioning
        if y is not None and self.num_classes is not None:
            if y.dim() == 1:
                # Class indices: [B]
                class_emb = self.label_emb(y)
            else:
                # Already embedded: [B, D]
                class_emb = y
            time_emb = time_emb + class_emb

        # Process context (text/crossattn conditioning)
        if context is not None and self.context_proj is not None:
            # context: [B, L, D] -> pool and project
            # Simple mean pooling for now
            context_pooled = context.mean(dim=1)  # [B, D]
            context_emb = self.context_proj(context_pooled)  # [B, model_channels]
        else:
            context_emb = None

        # Construct flow input
        flow_input_list = [x_flat, time_emb]
        if context_emb is not None:
            flow_input_list.append(context_emb)

        flow_input = torch.cat(flow_input_list, dim=-1)  # [B, flow_input_dim]

        # Forward through flow
        flow_output, log_det = self.flow.forward(flow_input)

        # Define energy function (key for exact gradient)
        # Energy depends on flow_output, which depends on x_flat
        energy = -torch.sum(flow_output ** 2, dim=-1)  # [B]

        # Compute EXACT score: ∇_{x_flat} energy
        score_flat = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=x_flat,
            create_graph=self.training,
            allow_unused=False,
        )[0]  # [B, C*H*W]

        # Enhance score through MLP (with all conditioning)
        projector_input_list = [score_flat, time_emb]
        if context_emb is not None:
            projector_input_list.append(context_emb)

        score_flat = self.score_projector(
            torch.cat(projector_input_list, dim=-1)
        )  # [B, C*H*W]

        # Reshape back to original shape
        if len(original_shape) > 2:
            score = score_flat.reshape(original_shape)
        else:
            score = score_flat

        # Compute F_θ to match denoiser expectation
        # Denoiser calls: network(x_noisy * c_in, c_noise, cond)
        # Expects: F_θ such that F_θ * c_out + x_noisy * c_skip = x_0

        # 1. Recover σ from c_noise (EDM: c_noise = 0.25 * log(σ))
        sigma = torch.exp(timesteps * 4.0)  # timesteps = c_noise = 0.25*log(σ)
        while sigma.dim() < x.dim():
            sigma = sigma.unsqueeze(-1)

        # 2. Compute EDM scaling coefficients
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2)**0.5
        c_in = 1.0 / (sigma**2 + self.sigma_data**2)**0.5

        # 3. Recover x_noisy from scaled input (x = x_noisy * c_in)
        x_noisy = x / c_in

        # 4. Compute x_0 from score: x_0 = x_noisy + σ² * score
        x_0 = x_noisy + (sigma**2) * score

        # 5. Compute F_θ = (x_0 - x_noisy * c_skip) / c_out
        F_theta = (x_0 - x_noisy * c_skip) / c_out

        return F_theta
