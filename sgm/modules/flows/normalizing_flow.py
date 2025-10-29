"""
Normalizing Flow implementation for score-based models
Ported and adapted from train_combined.py
"""

import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Tuple, Optional

from .flow_layers import FlowLayer, BaseActivation


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow that transforms a Gaussian base distribution
    to a target distribution through a sequence of invertible transformations.

    Can compute exact log probabilities and their gradients (score functions).
    """

    def __init__(
        self,
        dim: int,
        n_flows: int,
        activation: BaseActivation,
        base_mean: Optional[torch.Tensor] = None,
        base_std: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            dim: Dimensionality of the data
            n_flows: Number of flow layers
            activation: Activation function for flow layers
            base_mean: Mean of base Gaussian (default: zeros)
            base_std: Std of base Gaussian (default: ones)
        """
        super().__init__()
        self.dim = dim
        self.n_flows = n_flows

        # Base distribution: multivariate Gaussian
        if base_mean is None:
            base_mean = torch.zeros(dim)
        if base_std is None:
            base_cov = torch.eye(dim)
        else:
            base_cov = torch.diag(base_std ** 2)

        self.register_buffer('base_mean', base_mean)
        self.register_buffer('base_cov', base_cov)

        # Flow layers
        self.flows = nn.ModuleList([
            FlowLayer(dim, activation) for _ in range(n_flows)
        ])

        # Final MLP linear transformation (allows negative outputs)
        self.final_mlp = nn.Linear(dim, dim)
        self.final_mlp.bias.data.fill_(0)
        self.final_mlp.weight.data = torch.eye(dim) + 0.01 * torch.randn(dim, dim)

    @property
    def base_dist(self):
        """Dynamically create base distribution with correct device"""
        return dist.MultivariateNormal(self.base_mean, self.base_cov)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform base distribution samples to target distribution

        Args:
            z: Base distribution samples [batch_size, dim]

        Returns:
            x: Transformed samples [batch_size, dim]
            log_det_total: Total log determinant [batch_size]
        """
        log_det_total = torch.zeros(z.shape[0], device=z.device)

        x = z
        # Pass through flow layers
        for flow in self.flows:
            x, log_det = flow.forward(x)
            log_det_total += log_det

        # Final MLP linear transformation
        x = self.final_mlp(x)

        # MLP log determinant
        alpha = 1e-6
        regularized_weight = self.final_mlp.weight + alpha * torch.eye(
            self.dim, device=self.final_mlp.weight.device
        )
        mlp_log_det = torch.logdet(regularized_weight)

        if not torch.isfinite(mlp_log_det):
            mlp_log_det = torch.tensor(0.0, device=self.final_mlp.weight.device, requires_grad=True)

        log_det_total += mlp_log_det
        return x, log_det_total

    def log_prob(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of samples x given base samples z

        Args:
            x: Target samples [batch_size, dim]
            z: Base samples [batch_size, dim]

        Returns:
            log_prob: Log probability [batch_size]
        """
        log_prob_base = self.base_dist.log_prob(z)
        _, log_det_total = self.forward(z)
        log_prob_x = log_prob_base - log_det_total  # Change of variables formula
        return log_prob_x

    def sample(self, n_samples: int) -> torch.Tensor:
        """
        Generate samples from the learned distribution

        Args:
            n_samples: Number of samples to generate

        Returns:
            samples: Generated samples [n_samples, dim]
        """
        with torch.no_grad():
            z = self.base_dist.sample((n_samples,))
            if z.device != next(self.parameters()).device:
                z = z.to(next(self.parameters()).device)
            x, _ = self.forward(z)
        return x.detach()

    def compute_score(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute score function: ∇_x log p(x)

        Args:
            x: Target samples [batch_size, dim]
            z: Base samples [batch_size, dim]

        Returns:
            score: Score function ∇_x log p(x) [batch_size, dim]
        """
        x = x.requires_grad_(True)
        log_prob = self.log_prob(x, z)

        score = torch.autograd.grad(
            outputs=log_prob.sum(),
            inputs=x,
            create_graph=True
        )[0]

        return score
