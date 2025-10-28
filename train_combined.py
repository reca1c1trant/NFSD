"""
Combined Training Script for Normalizing Flow with Policy Gradient
All dependencies merged into a single file for easy deployment

This file combines:
- samplers.py
- activations.py
- flows_old.py
- objectives.py
- losses.py (modified to use flows_old instead of flows)
- train.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import copy
import argparse
import csv
import datetime
import time
from abc import ABC, abstractmethod
from typing import Tuple


# ==================== SAMPLERS ====================

class BaseSampler(ABC):
    @abstractmethod
    def sample(self, base_dist: dist.Distribution, n_samples: int) -> torch.Tensor:
        pass


class MonteCarloSampler(BaseSampler):
    def sample(self, base_dist: dist.Distribution, n_samples: int) -> torch.Tensor:
        return base_dist.sample((n_samples,))


# ==================== ACTIVATIONS ====================

class BaseActivation(ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        pass


class LeakyReLUActivation(BaseActivation):
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


class TrivialActivation(BaseActivation):
    def __init__(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def log_abs_det_jacobian(self, x: torch.Tensor):
        return torch.zeros(x.shape[:-1], device=x.device)


class SoftplusActivation(BaseActivation):
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


class Softplus2Activation(BaseActivation):
    def __init__(self, beta: float = 1.0):
        self.beta = beta
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.tensor(x>=0, dtype=torch.float32)
        y = mask * F.softplus(x, beta=self.beta) - (1-mask) * F.softplus(x, beta=self.beta)
        return y

    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        mask = torch.tensor(x>=0, dtype=torch.float32)
        log_mask_sigmoid = mask * -F.softplus(-x) - (1-mask) * (-F.softplus(-x))
        log_derivative = log_mask_sigmoid + torch.log(torch.tensor(self.beta, device=x.device))
        return torch.sum(log_derivative, dim=-1)


class ELUActivation(BaseActivation):
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


class GELUActivation(BaseActivation):
    """GELU activation: f(x) = x * Φ(x) where Φ is standard normal CDF"""
    def __init__(self):
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x)

    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        sqrt_2pi = torch.sqrt(torch.tensor(2.0 * 3.14159265359, device=x.device))
        phi_x = torch.exp(-0.5 * x**2) / sqrt_2pi
        Phi_x = 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device))))
        derivative = Phi_x + x * phi_x
        log_derivative = torch.log(torch.abs(derivative) + self.eps)
        return torch.sum(log_derivative, dim=-1)


class ShiftedSoftplusActivation(BaseActivation):
    """Shifted Softplus: f(x) = softplus(x) - offset"""
    def __init__(self, beta: float = 1.0, offset: float = 0.693147):
        self.beta = beta
        self.offset = offset
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(x, beta=self.beta) - self.offset

    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        beta_x = self.beta * x
        log_sigmoid = -F.softplus(-beta_x)
        log_derivative = log_sigmoid + torch.log(torch.tensor(self.beta, device=x.device))
        return torch.sum(log_derivative, dim=-1)


# ==================== OBJECTIVE FUNCTIONS ====================

def ellipsoid_function(x: torch.Tensor) -> torch.Tensor:
    """椭球函数: f(x) := ∑(i=1 to d) 10^(6(i-1)/(d-1)) * x_i^2"""
    d = x.shape[0]
    x = x + 0.2
    indices = torch.arange(d, dtype=torch.float32, device=x.device)
    coefficients = 10 ** ((6 * indices) / (d - 1))
    return torch.sum(x**2)


def discus_function(x: torch.Tensor) -> torch.Tensor:
    """盘状函数: f(x) := 10^6 * x_1^2 + ∑(i=2 to d) x_i^2"""
    x = x - 1
    first_term = 10**6 * x[0]**2
    rest_terms = torch.sum(x[1:]**2) if x.shape[0] > 1 else 0
    return first_term + rest_terms


def l1_ellipsoid_function(x: torch.Tensor) -> torch.Tensor:
    """L1椭球函数: f(x) := ∑(i=1 to d) 10^(6(i-1)/(d-1)) * |x_i|"""
    d = x.shape[0]
    x = x - 0.2
    indices = torch.arange(d, dtype=torch.float32, device=x.device)
    coefficients = 10 ** ((6 * indices) / (d - 1))
    return torch.sum(torch.abs(x))


def l_half_ellipsoid_function(x: torch.Tensor) -> torch.Tensor:
    """L1/2椭球函数: f(x) := ∑(i=1 to d) 10^(6(i-1)/(d-1)) * |x_i|^(1/2)"""
    d = x.shape[0]
    indices = torch.arange(d, dtype=torch.float32, device=x.device)
    coefficients = 10 ** ((6 * indices) / (d - 1))
    return torch.sum(coefficients * torch.abs(x)**0.5)


def levy_function(x: torch.Tensor) -> torch.Tensor:
    """Levy函数"""
    d = x.shape[0]
    x = x - 1
    if d == 1:
        w = 1 + (x - 1) / 4
        return torch.sin(torch.pi * w[0])**2

    w = 1 + (x - 1) / 4
    first_term = torch.sin(torch.pi * w[0])**2

    if d > 1:
        w_middle = w[:-1]
        middle_terms = (w_middle - 1)**2 * (1 + 10 * torch.sin(torch.pi * w_middle + 1)**2)
        middle_sum = torch.sum(middle_terms)
    else:
        middle_sum = 0

    last_term = (w[-1] - 1)**2 * (1 + torch.sin(2 * torch.pi * w[-1])**2)
    return first_term + middle_sum + last_term


def rastrigin10_function(x: torch.Tensor) -> torch.Tensor:
    """Rastrigin10函数"""
    d = x.shape[0]
    indices = torch.arange(d, dtype=torch.float32, device=x.device)
    scale_factors = 10 ** (indices / (d - 1))
    x = x - 1
    scaled_x = scale_factors * x
    quadratic_terms = scaled_x**2
    cosine_terms = 10 * torch.cos(2 * torch.pi * scaled_x)
    return 10 * d + torch.sum(quadratic_terms - cosine_terms)


# ==================== FLOWS ====================

class FlowLayer(nn.Module):
    def __init__(self, dim: int, activation: BaseActivation):
        super().__init__()
        self.dim = dim
        self.activation = activation

        # 下三角矩阵参数
        self.log_diagonal = nn.Parameter(torch.zeros(dim))
        self.lower_triangular = nn.Parameter(torch.zeros(dim, dim))
        self.bias = nn.Parameter(torch.zeros(dim))

        # 创建下三角mask
        self.register_buffer('tril_mask', torch.tril(torch.ones(dim, dim), diagonal=-1))

    def get_lower_triangular_matrix(self) -> torch.Tensor:
        """构造下三角矩阵，对角元素为正"""
        L = self.lower_triangular * self.tril_mask
        diag_mask = torch.eye(self.dim, device=L.device)
        L = L + diag_mask * torch.exp(self.log_diagonal)
        return L

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向变换: y = T(L @ x + b)"""
        batch_size = x.shape[0]
        L = self.get_lower_triangular_matrix()

        # 矩阵乘法 z = L @ x + b
        z = (L @ x.unsqueeze(-1)).squeeze(-1) + self.bias
        y = self.activation.forward(z)

        # 计算log determinant
        log_det_linear = torch.sum(self.log_diagonal)
        log_det_activation = self.activation.log_abs_det_jacobian(z)
        log_det = log_det_linear + log_det_activation

        return y, log_det


class NormalizingFlow(nn.Module):
    def __init__(self, dim: int, n_flows: int, activation: BaseActivation,
                 sampler: BaseSampler):
        super().__init__()
        self.dim = dim
        self.n_flows = n_flows
        self.sampler = sampler

        # 基础分布
        self.base_dist = dist.MultivariateNormal(torch.zeros(dim), torch.eye(dim))

        # Flow层
        self.flows = nn.ModuleList([
            FlowLayer(dim, activation) for _ in range(n_flows)
        ])

        # MLP线性变换层，允许负输出
        self.final_mlp = nn.Linear(dim, dim)
        self.final_mlp.bias.data.fill_(0)
        self.final_mlp.weight.data = torch.eye(dim) + 0.01 * torch.randn(dim, dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """将基础分布样本变换到目标分布"""
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        # 通过flow层
        for flow in self.flows:
            x, log_det = flow.forward(x)
            log_det_total += log_det

        # 最后通过MLP线性变换
        x = self.final_mlp(x)

        # MLP的log determinant
        alpha = 1e-6
        regularized_weight = self.final_mlp.weight + alpha * torch.eye(self.dim, device=self.final_mlp.weight.device)
        mlp_log_det = torch.logdet(regularized_weight)

        if not torch.isfinite(mlp_log_det):
            mlp_log_det = torch.tensor(0.0, device=self.final_mlp.weight.device, requires_grad=True)

        log_det_total += mlp_log_det
        return x, log_det_total

    def forward_with_base_samples(self, z_base: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用给定的base samples进行forward，返回samples和log_prob"""
        log_prob_base = self.base_dist.log_prob(z_base)
        log_det_total = torch.zeros(z_base.shape[0], device=z_base.device)
        x = z_base

        # 通过flow层
        for flow in self.flows:
            x, log_det = flow.forward(x)
            log_det_total = log_det_total + log_det

        # 最后通过MLP线性变换
        x = self.final_mlp(x)

        # MLP的log determinant
        alpha = 1e-6
        regularized_weight = self.final_mlp.weight + alpha * torch.eye(self.dim, device=self.final_mlp.weight.device)
        mlp_log_det = torch.logdet(regularized_weight)

        if not torch.isfinite(mlp_log_det):
            mlp_log_det = torch.tensor(0.0, device=self.final_mlp.weight.device, requires_grad=True)

        log_det_total += mlp_log_det
        log_prob_model = log_prob_base + log_det_total
        return x, log_prob_model

    def sample(self, n_samples: int) -> torch.Tensor:
        """仅生成样本，不计算概率"""
        with torch.no_grad():
            z = self.sampler.sample(self.base_dist, n_samples)
        x, _ = self.forward(z)
        return x.detach()

    def get_state_for_kl(self):
        """获取模型状态用于KL散度计算"""
        return {
            'state_dict': copy.deepcopy(self.state_dict()),
        }


# ==================== LOSS FUNCTION ====================

class PolicyGradientLoss:
    def __init__(self, kl_weight: float = 0.01, epsilon: float = 0.001):
        self.kl_weight = kl_weight
        self.epsilon = epsilon
        self.model_states_history = []
        self.fixed_base_samples = None

    def generate_new_base_samples(self, model: NormalizingFlow, n_samples: int):
        """生成新的固定base samples"""
        with torch.no_grad():
            self.fixed_base_samples = model.sampler.sample(model.base_dist, n_samples)

    def compute_loss(self, model: NormalizingFlow, target_func, epoch: int,
                    kl_update_interval: int):
        """计算损失：策略梯度损失 + KL散度损失"""
        if self.fixed_base_samples is None:
            raise ValueError("固定的base samples还没有生成！")

        # 使用固定的base samples计算forward
        samples, log_prob_model = model.forward_with_base_samples(self.fixed_base_samples)

        # 计算目标函数值
        with torch.no_grad():
            target_values = torch.stack([target_func(sample) for sample in samples])
            min_target_value = target_values.min()
            max_target_value = target_values.max()
            target_values_norm = (target_values - min_target_value) / ((max_target_value - min_target_value) + 1e-6)

        mean_target_value = target_values.mean()

        # KL散度损失
        kl_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        if len(self.model_states_history) != 0:
            # 获取前一个模型并计算其log_prob
            prev_state = self.model_states_history[-1]
            prev_model = copy.deepcopy(model)
            prev_model.load_state_dict(prev_state['state_dict'])
            prev_model.eval()

            with torch.no_grad():
                _, prev_log_prob = prev_model.forward_with_base_samples(self.fixed_base_samples)

            # KL散度损失：E[log p_current(x) - log p_previous(x)]
            kl_loss = torch.mean(log_prob_model - prev_log_prob)

        # 策略梯度损失
        policy_grad_loss = torch.mean(target_values * log_prob_model)

        # 熵损失
        entropy_loss = torch.mean(log_prob_model)

        # 总损失
        total_loss = policy_grad_loss + self.kl_weight * kl_loss

        return {
            'total_loss': total_loss,
            'policy_grad_loss': policy_grad_loss,
            'kl_loss': kl_loss,
            'mean_target_value': mean_target_value,
            'min_target_value': min_target_value,
            'entropy_loss': entropy_loss,
        }

    def save_current_model_state(self, model: NormalizingFlow, epoch: int):
        """保存当前epoch的模型状态 - 队列行为，只保持最近1个状态"""
        current_state = model.get_state_for_kl()
        current_state['epoch'] = epoch
        self.model_states_history.append(current_state)

        # 队列行为：只保持最近1个状态
        if len(self.model_states_history) > 1:
            self.model_states_history.pop(0)


# ==================== TRAINING FUNCTIONS ====================

def get_target_function(func_name):
    """根据缩写名称获取目标函数"""
    functions = {
        'l1': l1_ellipsoid_function,
        'lhalf': l_half_ellipsoid_function,
        'levy': levy_function,
        'rastrigin': rastrigin10_function,
        'discus': discus_function,
        'ellipsoid': ellipsoid_function,
        'ell': ellipsoid_function,
        'ras': rastrigin10_function,
    }
    return functions.get(func_name, l1_ellipsoid_function)


def get_function_display_name(func_name):
    """获取函数的显示名称"""
    display_names = {
        'l1': 'L1椭球',
        'lhalf': 'L1/2椭球',
        'levy': 'Levy',
        'rastrigin': 'Rastrigin10',
        'discus': '盘状',
        'ellipsoid': '椭球',
        'ell': '椭球',
        'ras': 'Rastrigin10',
    }
    return display_names.get(func_name, func_name)


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Normalizing Flow with Policy Gradient')

    # 模型参数
    parser.add_argument('--dim', type=int, default=1000, help='Dimension of the problem')
    parser.add_argument('--n_flows', type=int, default=2, help='Number of flow layers')
    parser.add_argument('--activation', type=str, default='softplus',
                       choices=['softplus', 'leakyrelu', 'elu', 'tanh', "softplus2"],
                       help='Activation function')

    # 目标函数选择
    parser.add_argument('--func', type=str, default='l1',
                       choices=['l1', 'lhalf', 'levy', 'rastrigin', 'discus', 'ellipsoid', 'ell', 'ras'],
                       help='Target function to optimize')

    # 训练参数
    parser.add_argument('--n_epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for policy gradient')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

    # KL散度参数
    parser.add_argument('--kl_weight', type=float, default=0.01, help='Weight for KL regularization')
    parser.add_argument('--update_interval', type=int, default=200000,
                       help='Interval (epochs) to update fixed base samples')

    # 激活函数参数
    parser.add_argument('--softplus_beta', type=float, default=1, help='Beta parameter for softplus')
    parser.add_argument('--leakyrelu_slope', type=float, default=10, help='Negative slope for LeakyReLU')
    parser.add_argument('--elu_alpha', type=float, default=1.0, help='Alpha parameter for ELU')

    return parser.parse_args()


def create_activation(args):
    """根据参数创建激活函数"""
    if args.activation == 'softplus':
        return SoftplusActivation(beta=args.softplus_beta)
    elif args.activation == 'leakyrelu':
        return LeakyReLUActivation(negative_slope=args.leakyrelu_slope)
    elif args.activation == 'elu':
        return ELUActivation(alpha=args.elu_alpha)
    elif args.activation == 'tanh':
        return TanhActivation()
    elif args.activation == 'softplus2':
        return Softplus2Activation(beta=args.softplus_beta)
    else:
        raise ValueError(f"Unknown activation: {args.activation}")


def train_normalizing_flow_with_policy_gradient(target_func, args):
    """使用策略梯度训练normalizing flow"""

    # 模型初始化
    activation = create_activation(args)
    sampler = MonteCarloSampler()
    model = NormalizingFlow(args.dim, args.n_flows, activation, sampler)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 损失函数
    loss_fn = PolicyGradientLoss(kl_weight=args.kl_weight, epsilon=0.001)

    # 监控数据收集
    monitoring_data = []

    # best value跟踪
    best_value = float('inf')
    best_sample = None

    start_time = time.time()

    for epoch in range(args.n_epochs):
        # 每隔指定epochs重新生成固定的base samples
        if epoch % args.update_interval == 0:
            loss_fn.generate_new_base_samples(model, args.batch_size)

        optimizer.zero_grad()

        # 计算损失（策略梯度 + KL散度）
        loss_dict = loss_fn.compute_loss(model, target_func, epoch, args.update_interval)
        total_loss = loss_dict['total_loss']

        # 反向传播
        total_loss.backward()
        optimizer.step()

        # 在每个epoch结束后保存当前模型状态
        loss_fn.save_current_model_state(model, epoch)

        # 更新best value
        current_min = loss_dict['min_target_value'].item()
        if current_min < best_value:
            best_value = current_min
            # 生成样本来获取best sample
            with torch.no_grad():
                temp_samples = model.sample(args.batch_size)
                temp_values = torch.stack([target_func(s) for s in temp_samples])
                best_idx = torch.argmin(temp_values)
                best_sample = temp_samples[best_idx].clone()

        if epoch % 10 == 0:
            # 收集监控数据到字典
            monitor_record = {
                'epoch': epoch,
                'grad_loss': loss_dict['policy_grad_loss'].item(),
                'kl_loss': loss_dict['kl_loss'].item(),
                'total_loss': total_loss.item(),
                'mean_target_value': loss_dict['mean_target_value'].item(),
                'min_target_value': loss_dict['min_target_value'].item(),
                'entropy_loss': loss_dict['entropy_loss'].item()
            }
            monitoring_data.append(monitor_record)

            # 打印进度
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d}: Total={total_loss.item():8.4f}, Mean={loss_dict['mean_target_value'].item():8.4f}, "
                  f"Min={loss_dict['min_target_value'].item():8.4f}, Best={best_value:8.4f}, "
                  f"Grad={loss_dict['policy_grad_loss'].item():6.3f}, KL={loss_dict['kl_loss'].item():6.3f}, "
                  f"Entropy={loss_dict['entropy_loss'].item():6.3f}, Time={elapsed:.1f}s")

    # 保存监控数据为CSV
    if monitoring_data:
        csv_filename = f"training_monitor_dim{args.dim}_flows{args.n_flows}_{args.activation}_{args.func}.csv"

        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['epoch', 'grad_loss', 'kl_loss', 'total_loss',
                         'mean_target_value', 'min_target_value', 'entropy_loss']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for record in monitoring_data:
                writer.writerow(record)

        print(f"\n监控数据已保存到: {csv_filename}")
        print(f"总共记录了 {len(monitoring_data)} 个监控点")

    # 最终结果打印
    total_time = time.time() - start_time
    print(f"\n=== Training Completed ===")
    print(f"Training completed in {total_time:.2f}s")
    print(f"Final best value: {best_value:.8f}")
    print(f"Final best sample: {best_sample}")

    return model, best_value, best_sample, total_time


# ==================== MAIN ====================

if __name__ == "__main__":
    # 解析参数
    args = get_args()

    # 获取选择的目标函数
    target_func = get_target_function(args.func)
    func_display_name = get_function_display_name(args.func)

    print("=== Normalizing Flow 训练参数 ===")
    print(f"目标函数: {func_display_name} ({args.func})")
    print(f"维度: {args.dim}")
    print(f"Flow层数: {args.n_flows}")
    print(f"激活函数: {args.activation}")
    print(f"训练轮数: {args.n_epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print(f"KL权重: {args.kl_weight}")
    print()

    print("其他参数:")
    for arg, value in vars(args).items():
        if arg not in ['func', 'dim', 'n_flows', 'activation', 'n_epochs', 'batch_size', 'lr', 'kl_weight']:
            print(f"  {arg}: {value}")
    print()

    # 训练模型
    print(f"开始训练 {func_display_name} 函数优化...")
    trained_model, best_value, best_sample, training_time = train_normalizing_flow_with_policy_gradient(target_func, args)

    # 生成样本
    print(f"\n=== 模型评估 ===")
    samples = trained_model.sample(1000)
    print(f"Generated samples shape: {samples.shape}")

    # 评估样本质量
    with torch.no_grad():
        sample_values = torch.stack([target_func(s) for s in samples])
        print(f"Sample function values:")
        print(f"  Mean: {sample_values.mean():.6f}")
        print(f"  Std: {sample_values.std():.6f}")
        print(f"  Min: {sample_values.min():.6f}")
        print(f"  Max: {sample_values.max():.6f}")
        print(f"  Best sample: {best_sample}")
        print(f"  Best value: {best_value:.8f}")

    # 记录到output.txt
    with open('output.txt', 'a', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NORMALIZING FLOW TRAINING RESULTS (COMBINED)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script: train_combined.py\n\n")

        # 超参数记录
        f.write("=== Hyperparameters ===\n")
        f.write(f"Target Function: {func_display_name} ({args.func})\n")
        f.write(f"Dimension: {args.dim}\n")
        f.write(f"Flow layers: {args.n_flows}\n")
        f.write(f"Activation: {args.activation}\n")
        f.write(f"Training epochs: {args.n_epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"KL weight: {args.kl_weight}\n")
        f.write(f"Update interval: {args.update_interval}\n")

        # 激活函数参数
        if args.activation == 'softplus':
            f.write(f"Softplus beta: {args.softplus_beta}\n")
        elif args.activation == 'leakyrelu':
            f.write(f"LeakyReLU slope: {args.leakyrelu_slope}\n")
        elif args.activation == 'elu':
            f.write(f"ELU alpha: {args.elu_alpha}\n")
        f.write("\n")

        # 训练结果
        f.write("=== Training Results ===\n")
        f.write(f"Training completed in {training_time:.2f}s\n")
        f.write(f"Best value found: {best_value:.8f}\n")
        f.write(f"Best sample: {best_sample.tolist()}\n\n")

        # 模型评估结果
        f.write("=== Model Evaluation ===\n")
        f.write(f"Generated samples shape: {samples.shape}\n")
        f.write(f"Sample function values:\n")
        f.write(f"  Mean: {sample_values.mean():.6f}\n")
        f.write(f"  Std: {sample_values.std():.6f}\n")
        f.write(f"  Min: {sample_values.min():.6f}\n")
        f.write(f"  Max: {sample_values.max():.6f}\n")
        f.write(f"  Best sample: {best_sample.tolist()}\n")
        f.write(f"  Best value: {best_value:.8f}\n")

        # CSV文件信息
        csv_filename = f"training_monitor_dim{args.dim}_flows{args.n_flows}_{args.activation}_{args.func}.csv"
        f.write(f"CSV file: {csv_filename}\n")
        f.write("\n")
        f.write("\n")

    print(f"结果已记录到 output.txt")
