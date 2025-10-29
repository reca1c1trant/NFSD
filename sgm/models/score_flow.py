"""
Score-based Diffusion Model using Normalizing Flow
Replaces UNet + Denoiser with Flow-based score computation
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional, Union, List, Tuple
from omegaconf import ListConfig, OmegaConf

from ..modules.flows import NormalizingFlow
from ..modules.flows.flow_layers import get_activation
from ..util import instantiate_from_config, default, get_obj_from_str
from ..modules import UNCONDITIONAL_CONFIG


class ScoreFlowNetwork(nn.Module):
    """
    Score network based on Normalizing Flow
    Computes score function: ∇_x log p_t(x_t)
    """

    def __init__(
        self,
        data_dim: int,
        n_flows: int = 3,
        activation: str = 'softplus',
        activation_params: Optional[Dict] = None,
        sigma_embed_dim: int = 256,
        cond_embed_dim: int = 512,
        use_conditioning: bool = True,
    ):
        """
        Args:
            data_dim: Dimension of the data (e.g., flattened latent size)
            n_flows: Number of flow layers
            activation: Activation function name
            activation_params: Parameters for activation function
            sigma_embed_dim: Dimension of sigma embedding
            cond_embed_dim: Dimension of conditional embedding
            use_conditioning: Whether to use conditioning
        """
        super().__init__()
        self.data_dim = data_dim
        self.use_conditioning = use_conditioning

        # Sigma (noise level) embedder
        self.sigma_embedder = nn.Sequential(
            nn.Linear(1, sigma_embed_dim),
            nn.SiLU(),
            nn.Linear(sigma_embed_dim, sigma_embed_dim),
            nn.SiLU(),
        )

        # Calculate flow input dimension
        flow_input_dim = data_dim + sigma_embed_dim
        if use_conditioning:
            flow_input_dim += cond_embed_dim
            # Conditional embedding projector (will be fed external embeddings)
            self.cond_projector = nn.Linear(cond_embed_dim, cond_embed_dim)

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

        # Output projector: project back to data dimension for score
        self.score_projector = nn.Sequential(
            nn.Linear(flow_input_dim, data_dim * 2),
            nn.SiLU(),
            nn.Linear(data_dim * 2, data_dim),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        sigma_t: torch.Tensor,
        cond_embedding: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute score function and predicted clean sample using exact gradient

        Args:
            x_t: Noisy samples [B, D] or [B, C, H, W]
            sigma_t: Noise levels [B] or [B, 1]
            cond_embedding: Conditional embeddings [B, cond_dim] (optional)

        Returns:
            score: Score function ∇ log p_t(x_t) [B, D] or [B, C, H, W]
            x_pred: Predicted clean sample [B, D] or [B, C, H, W]
        """
        batch_size = x_t.shape[0]
        original_shape = x_t.shape

        # Flatten if needed
        if len(x_t.shape) > 2:
            x_flat = x_t.reshape(batch_size, -1)
        else:
            x_flat = x_t

        # Enable gradient computation for x_flat
        x_flat = x_flat.requires_grad_(True)

        # Embed sigma
        if sigma_t.dim() == 1:
            sigma_t = sigma_t.unsqueeze(-1)
        sigma_emb = self.sigma_embedder(sigma_t)  # [B, sigma_embed_dim]

        # Prepare conditioning
        if self.use_conditioning and cond_embedding is not None:
            cond_emb = self.cond_projector(cond_embedding)
        else:
            cond_emb = None

        # Compute energy-like function via Flow
        # Flow takes x_t and outputs transformation
        flow_input_list = [x_flat, sigma_emb]
        if cond_emb is not None:
            flow_input_list.append(cond_emb)

        flow_input = torch.cat(flow_input_list, dim=-1)

        # Sample from base distribution (fixed for this batch)
        with torch.no_grad():
            z_base = torch.randn(batch_size, flow_input.size(-1), device=x_flat.device)

        # Forward through flow
        flow_output, log_det = self.flow.forward(flow_input)

        # Compute log probability (energy function)
        # This is the key: flow_output depends on x_flat through flow_input
        log_prob_base = self.flow.base_dist.log_prob(z_base)

        # Energy: negative log likelihood
        # We use the flow transformation as an energy-based model
        energy = -torch.sum(flow_output ** 2, dim=-1)  # Simple energy function

        # Compute score: ∇_{x_flat} energy
        score_flat = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=x_flat,
            create_graph=self.training,
            allow_unused=False,
        )[0]

        # Project score through MLP for better expressiveness
        # Include all conditioning information to match flow_input_dim
        projector_input_list = [score_flat, sigma_emb]
        if cond_emb is not None:
            projector_input_list.append(cond_emb)

        score_flat = self.score_projector(
            torch.cat(projector_input_list, dim=-1)
        )

        # Reshape back
        if len(original_shape) > 2:
            score = score_flat.reshape(original_shape)
        else:
            score = score_flat

        # Compute predicted clean sample from score
        # Using: ∇ log p_t(x_t) = (x_0 - x_t) / σ_t²
        # Therefore: x_0 = x_t + σ_t² * score
        sigma_t_expanded = sigma_t.squeeze(-1)
        while sigma_t_expanded.dim() < score.dim():
            sigma_t_expanded = sigma_t_expanded.unsqueeze(-1)

        x_pred = x_t.detach() + (sigma_t_expanded ** 2) * score

        return score, x_pred


class FlowDiffusionEngine(pl.LightningModule):
    """
    Diffusion model using Flow-based score network
    Compatible with the existing training infrastructure
    """

    def __init__(
        self,
        score_network_config,
        conditioner_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        first_stage_config=None,
        loss_fn_config=None,
        optimizer_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        scheduler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        sampler_config: Union[None, Dict, ListConfig, OmegaConf] = None,
        input_key: str = "jpg",
        log_keys: Union[List, None] = None,
        scale_factor: float = 1.0,
        disable_first_stage_autocast: bool = False,
        train_first_stage: bool = False,  # NEW: whether to train autoencoder
    ):
        super().__init__()
        self.input_key = input_key
        self.log_keys = log_keys
        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.train_first_stage = train_first_stage

        # Score network (replaces UNet)
        self.score_network = instantiate_from_config(score_network_config)

        # Conditioner (same as original)
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )

        # First stage model (VAE/Autoencoder)
        self.first_stage_model = instantiate_from_config(first_stage_config)

        # Control whether to train first_stage_model
        if not train_first_stage:
            self.first_stage_model.eval()
            self.first_stage_model.train = lambda *args, **kwargs: None
            for param in self.first_stage_model.parameters():
                param.requires_grad = False

        # Loss function
        self.loss_fn = instantiate_from_config(loss_fn_config) if loss_fn_config else None

        # Optimizer config
        self.optimizer_config = default(
            optimizer_config, {"target": "torch.optim.AdamW"}
        )

        # Scheduler config
        self.scheduler_config = scheduler_config

        # Sampler config (for inference)
        self.sampler_config = sampler_config

    def encode_first_stage(self, x):
        """Encode images to latent space"""
        with torch.no_grad():
            z = self.first_stage_model.encode(x)
            z = self.scale_factor * z
        return z

    def decode_first_stage(self, z):
        """Decode latents to image space"""
        z = z / self.scale_factor
        with torch.no_grad():
            x = self.first_stage_model.decode(z)
        return x

    def forward(self, x, sigma, cond):
        """Forward pass through score network"""
        return self.score_network(x, sigma, cond)

    def shared_step(self, batch: Dict) -> torch.Tensor:
        """Shared training/validation step"""
        # Get input
        x = batch[self.input_key]

        # Encode to latent space if using VAE
        if hasattr(self, 'first_stage_model'):
            x = self.encode_first_stage(x)

        batch_size = x.shape[0]

        # Sample noise level
        if self.loss_fn and hasattr(self.loss_fn, 'sigma_sampler'):
            sigma = self.loss_fn.sigma_sampler(batch_size).to(x.device)
        else:
            # Default: uniform sampling in log space
            log_sigma = torch.rand(batch_size, device=x.device) * 2 - 1  # [-1, 1]
            sigma = torch.exp(log_sigma)

        # Add noise
        noise = torch.randn_like(x)
        x_noisy = x + sigma.view(batch_size, *([1] * (x.ndim - 1))) * noise

        # Get conditioning
        cond_dict = self.conditioner(batch)
        # Flatten conditioning to single embedding
        if isinstance(cond_dict, dict):
            cond_emb = torch.cat([v for v in cond_dict.values() if isinstance(v, torch.Tensor)], dim=-1)
        else:
            cond_emb = None

        # Predict
        score, x_pred = self.score_network(x_noisy, sigma, cond_emb)

        # Compute loss
        if self.loss_fn:
            loss = self.loss_fn(x_pred, x, sigma)
        else:
            # Default MSE loss
            loss = torch.mean((x_pred - x) ** 2)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        # 获取学习率和其他参数
        optimizer_params_config = self.optimizer_config.get("params", {})
        lr = optimizer_params_config.get("lr", 1e-4)

        # 获取 optimizer 类
        optimizer_class = get_obj_from_str(self.optimizer_config["target"])

        # 准备 optimizer 参数（排除 lr）
        optimizer_kwargs = {
            k: v for k, v in optimizer_params_config.items()
            if k != "lr"
        }

        # Collect parameters to optimize
        params_to_optimize = list(self.score_network.parameters())

        # Add first_stage_model parameters if training it
        if self.train_first_stage:
            params_to_optimize += list(self.first_stage_model.parameters())

        # 创建 optimizer
        optimizer = optimizer_class(
            params_to_optimize,
            lr=lr,
            **optimizer_kwargs
        )

        # 如果有 scheduler 配置
        if self.scheduler_config is not None:
            scheduler_class = get_obj_from_str(self.scheduler_config["target"])
            scheduler_params = self.scheduler_config.get("params", {})
            scheduler = scheduler_class(optimizer, **scheduler_params)
            return [optimizer], [scheduler]

        return optimizer

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_steps: int = 50,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        cond: Optional[Dict] = None,
    ):
        """
        Sample from the model using ancestral sampling

        Args:
            batch_size: Number of samples
            num_steps: Number of denoising steps
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            cond: Conditioning information

        Returns:
            samples: Generated samples
        """
        # Initialize from noise
        device = next(self.parameters()).device
        shape = (batch_size, self.score_network.data_dim)
        x = torch.randn(shape, device=device) * sigma_max

        # Get conditioning embedding
        if cond is not None:
            cond_dict = self.conditioner(cond)
            cond_emb = torch.cat([v for v in cond_dict.values() if isinstance(v, torch.Tensor)], dim=-1)
        else:
            cond_emb = None

        # Denoising loop
        sigmas = torch.linspace(sigma_max, sigma_min, num_steps, device=device)

        for i in range(num_steps):
            sigma = sigmas[i].expand(batch_size)

            # Compute score and prediction
            score, x_pred = self.score_network(x, sigma, cond_emb)

            if i < num_steps - 1:
                # Ancestral sampling step
                sigma_next = sigmas[i + 1]
                x = x + (sigma - sigma_next) * score
                if sigma_next > 0:
                    x = x + torch.randn_like(x) * torch.sqrt(sigma_next ** 2 - sigma ** 2)
            else:
                # Final step: use prediction directly
                x = x_pred

        return x