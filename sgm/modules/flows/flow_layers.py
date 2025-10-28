"""
Flow layer implementations with various activation functions
Ported from train_combined.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple


class BaseActivation(ABC):
    """Base class for activation functions with log-det-jacobian"""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        pass


class TrivialActivation(BaseActivation):
    """Identity activation (no transformation)"""

    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def log_abs_det_jacobian(self, x: torch.Tensor):
        return torch.zeros(x.shape[:-1], device=x.device)


class LeakyReLUActivation(BaseActivation):
    """LeakyReLU activation with log-det-jacobian"""

    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope)

    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        log_slope = torch.log(torch.tensor(self.negative_slope, device=x.device))
        elementwise_log_det = torch.where(x >= 0,
                                        torch.zeros_like(x),
                                        torch.full_like(x, log_slope))
        return torch.sum(elementwise_log_det, dim=-1)


class SoftplusActivation(BaseActivation):
    """Softplus activation with log-det-jacobian"""

    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, beta=self.beta)

    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        beta_x = self.beta * x
        log_sigmoid = -F.softplus(-beta_x)
        log_derivative = log_sigmoid + torch.log(torch.tensor(self.beta, device=x.device))
        return torch.sum(log_derivative, dim=-1)


class ELUActivation(BaseActivation):
    """ELU activation with log-det-jacobian"""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, alpha=self.alpha)

    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        elementwise_log_det = torch.where(
            x >= 0,
            torch.zeros_like(x),
            torch.log(torch.tensor(self.alpha, device=x.device)) + x
        )
        return torch.sum(elementwise_log_det, dim=-1)


class TanhActivation(BaseActivation):
    """Tanh activation with log-det-jacobian"""

    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        abs_x = torch.abs(x)
        log_cosh = abs_x + torch.log(1 + torch.exp(-2 * abs_x)) - torch.log(torch.tensor(2.0, device=x.device))
        elementwise_log_det = -2 * log_cosh
        return torch.sum(elementwise_log_det, dim=-1)


class SwishActivation(BaseActivation):
    """Swish activation: f(x) = x * sigmoid(x)"""

    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.beta * x)

    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        beta_x = self.beta * x
        sigmoid_beta_x = torch.sigmoid(beta_x)
        derivative = sigmoid_beta_x * (1 + beta_x * (1 - sigmoid_beta_x))
        log_derivative = torch.log(torch.abs(derivative) + self.eps)
        return torch.sum(log_derivative, dim=-1)


class FlowLayer(nn.Module):
    """
    Single flow layer: affine transformation + nonlinear activation
    y = activation(L @ x + b)
    where L is lower triangular with positive diagonal
    """

    def __init__(self, dim: int, activation: BaseActivation):
        super().__init__()
        self.dim = dim
        self.activation = activation

        # Lower triangular matrix parameters
        self.log_diagonal = nn.Parameter(torch.zeros(dim))
        self.lower_triangular = nn.Parameter(torch.zeros(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

        # Create lower triangular mask
        self.register_buffer('tril_mask', torch.tril(torch.ones(dim, dim), diagonal=-1))

    def get_lower_triangular_matrix(self) -> torch.Tensor:
        """Construct lower triangular matrix with positive diagonal"""
        L = self.lower_triangular * self.tril_mask
        diag_mask = torch.eye(self.dim, device=L.device)
        L = L + diag_mask * torch.exp(self.log_diagonal)
        return L

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation: y = activation(L @ x + b)

        Args:
            x: Input tensor [batch_size, dim]

        Returns:
            y: Transformed tensor [batch_size, dim]
            log_det: Log determinant of Jacobian [batch_size]
        """
        batch_size = x.shape[0]
        L = self.get_lower_triangular_matrix()

        # Matrix multiplication z = L @ x + b
        z = (L @ x.unsqueeze(-1)).squeeze(-1) + self.bias
        y = self.activation.forward(z)

        # Compute log determinant
        log_det_linear = torch.sum(self.log_diagonal)
        log_det_activation = self.activation.log_abs_det_jacobian(z)
        log_det = log_det_linear + log_det_activation

        return y, log_det


def get_activation(activation_name: str, **kwargs) -> BaseActivation:
    """Factory function to create activation by name"""
    activations = {
        'trivial': TrivialActivation,
        'leakyrelu': LeakyReLUActivation,
        'softplus': SoftplusActivation,
        'elu': ELUActivation,
        'tanh': TanhActivation,
        'swish': SwishActivation,
    }

    if activation_name not in activations:
        raise ValueError(f"Unknown activation: {activation_name}")

    return activations[activation_name](**kwargs)
