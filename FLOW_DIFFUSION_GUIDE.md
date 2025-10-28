# Flow-based Diffusion Model 使用指南

## 📋 概述

本实现用 **Normalizing Flow** 替代了传统 Stable Diffusion 的 **UNet + Denoiser** 架构，通过精确的概率建模直接计算 score function。

### 核心创新

```
原架构: UNet(x_t, σ_t) → x̂_θ → score = (x̂_θ - x_t)/σ_t²
新架构: NormalizingFlow(x_t) → ∇log p(x_t) = score → x̂_θ = x_t + σ_t²·score
```

---

## 📁 新增文件清单

### 核心模块
```
sgm/modules/flows/
├── __init__.py                  # Flow 模块入口
├── flow_layers.py               # Flow 层和激活函数
└── normalizing_flow.py          # Normalizing Flow 实现

sgm/models/
└── score_flow.py                # ScoreFlowNetwork 和 FlowDiffusionEngine
```

### 配置文件
```
configs/training/
├── flow_diffusion_mnist.yaml    # MNIST 训练配置
└── flow_diffusion_toy.yaml      # 玩具数据配置
```

### 训练和采样脚本
```
scripts/
├── train_flow_diffusion.py                  # 训练脚本
└── sampling/
    └── simple_flow_sample.py                # 采样脚本
```

---

## 🚀 快速开始

### 1. 环境准备

确保已按照 CLAUDE.md 完成环境安装：

```bash
# 激活虚拟环境
source .pt2/bin/activate  # Linux/Mac
# 或
.pt2\Scripts\activate     # Windows

# 确保已安装项目
pip install -e .
```

### 2. 训练模型

#### 在 MNIST 上训练

```bash
python scripts/train_flow_diffusion.py \
    --config configs/training/flow_diffusion_mnist.yaml \
    --name mnist_flow_test \
    --seed 42
```

#### 使用 W&B 记录

```bash
python scripts/train_flow_diffusion.py \
    --config configs/training/flow_diffusion_mnist.yaml \
    --name mnist_flow_wandb \
    --use_wandb \
    --wandb_project my-flow-diffusion
```

#### 从检查点恢复训练

```bash
python scripts/train_flow_diffusion.py \
    --config configs/training/flow_diffusion_mnist.yaml \
    --resume logs/2024-xx-xx-name/checkpoints/last.ckpt
```

### 3. 生成样本

```bash
python scripts/sampling/simple_flow_sample.py \
    --config configs/training/flow_diffusion_mnist.yaml \
    --checkpoint logs/2024-xx-xx-mnist_flow_test/checkpoints/last.ckpt \
    --num_samples 16 \
    --num_steps 50 \
    --output_dir outputs/flow_mnist
```

#### 条件生成（指定类别）

```bash
python scripts/sampling/simple_flow_sample.py \
    --config configs/training/flow_diffusion_mnist.yaml \
    --checkpoint logs/.../checkpoints/last.ckpt \
    --num_samples 16 \
    --class_label 3 \
    --output_dir outputs/flow_mnist_class3
```

---

## ⚙️ 配置说明

### 关键参数

#### ScoreFlowNetwork 参数

```yaml
score_network_config:
  params:
    data_dim: 784              # 数据维度（MNIST: 28*28=784）
    n_flows: 3                 # Flow 层数（越多表达能力越强）
    activation: softplus       # 激活函数（softplus/leakyrelu/elu/tanh）
    activation_params:
      beta: 1.0                # Softplus 的 beta 参数
    sigma_embed_dim: 256       # 噪声水平嵌入维度
    cond_embed_dim: 128        # 条件嵌入维度
    use_conditioning: true     # 是否使用条件生成
```

#### 训练参数

```yaml
model:
  base_learning_rate: 1.0e-4   # 学习率

data:
  params:
    batch_size: 256            # 批次大小
    num_workers: 4             # 数据加载线程数

lightning:
  trainer:
    max_epochs: 50             # 训练轮数
    gradient_clip_val: 1.0     # 梯度裁剪
```

---

## 🔧 高级用法

### 自定义激活函数

在 `sgm/modules/flows/flow_layers.py` 中添加新的激活函数：

```python
class MyActivation(BaseActivation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 你的激活函数
        return ...

    def log_abs_det_jacobian(self, x: torch.Tensor) -> torch.Tensor:
        # 计算 log |det(Jacobian)|
        return ...
```

然后在 `get_activation()` 中注册：

```python
activations = {
    ...
    'myactivation': MyActivation,
}
```

### 修改 Flow 层数

编辑配置文件中的 `n_flows` 参数：

```yaml
score_network_config:
  params:
    n_flows: 5  # 增加到 5 层
```

**注意**：层数越多，模型表达能力越强，但训练时间也越长。

### 调整噪声调度

修改 `sigma_sampler_config`：

```yaml
loss_fn_config:
  params:
    sigma_sampler_config:
      target: sgm.modules.diffusionmodules.sigma_sampling.EDMSampling
      params:
        p_mean: -1.2    # 控制噪声分布的中心
        p_std: 1.2      # 控制噪声分布的范围
```

---

## 📊 架构对比

| 组件 | 原 Stable Diffusion | Flow Diffusion |
|------|---------------------|----------------|
| **主干网络** | UNet (~2.6B 参数) | NormalizingFlow (~数百万参数) |
| **Score 计算** | 间接（通过 x̂ 推导） | 直接（梯度计算） |
| **Denoiser** | 需要单独模块 | 不需要（公式直接推导） |
| **理论基础** | 经验性强 | 精确概率建模 |
| **训练稳定性** | 需要大量数据 | 相对简单 |

---

## 🐛 常见问题

### 1. 内存不足

**问题**：训练时 CUDA out of memory

**解决**：
- 减小 `batch_size`
- 减少 `n_flows`
- 减小 `data_dim`（使用 VAE latent space）

### 2. 训练不收敛

**问题**：Loss 不下降或不稳定

**解决**：
- 降低学习率（如 `1e-5`）
- 启用梯度裁剪：`gradient_clip_val: 1.0`
- 调整 `sigma_sampler` 的参数
- 增加 `n_flows` 提高表达能力

### 3. 生成质量差

**问题**：采样结果模糊或质量低

**解决**：
- 增加采样步数 `--num_steps 100`
- 调整 `sigma_min` 和 `sigma_max`
- 训练更多 epoch
- 检查数据预处理（归一化）

### 4. 梯度消失/爆炸

**问题**：训练过程中梯度异常

**解决**：
- 使用不同的激活函数（试试 `elu` 或 `swish`）
- 启用梯度裁剪
- 减小学习率

---

## 📈 扩展到更复杂数据

### 在 CIFAR-10 上训练

1. 创建配置文件 `configs/training/flow_diffusion_cifar.yaml`
2. 修改 `data_dim: 3072`（32×32×3）
3. 增加 `n_flows: 5` 或更多
4. 使用 CIFAR-10 数据加载器

### 在 Latent Space 上训练（类似 SDXL）

1. 使用 VAE 编码器
2. 配置 `first_stage_config` 指向预训练的 VAE
3. 设置 `data_dim` 为 latent 维度（如 SDXL 的 4×64×64 = 16384）
4. 增加模型容量

---

## 🔬 实验建议

### 消融实验

1. **Flow 层数影响**：测试 `n_flows = 1, 2, 3, 5, 10`
2. **激活函数比较**：对比 `softplus`, `elu`, `swish`, `tanh`
3. **条件 vs 无条件**：测试 `use_conditioning: true/false`
4. **采样步数**：测试 `num_steps = 10, 20, 50, 100`

### 性能基准

运行以下命令进行基准测试：

```bash
# 训练速度
time python scripts/train_flow_diffusion.py --config configs/training/flow_diffusion_mnist.yaml

# 采样速度
time python scripts/sampling/simple_flow_sample.py --config ... --num_samples 100
```

---

## 📚 技术细节

### Score Function 计算

在 `ScoreFlowNetwork.forward()` 中：

```python
# 1. 构造输入：[x_t, σ_t, conditions]
flow_input = cat([x_t, sigma_emb, cond_emb])

# 2. 通过 Flow 得到 log p(x)
log_prob = flow.log_prob(flow_input)

# 3. 计算梯度（score）
score = ∇_{x_t} log_prob

# 4. 恢复干净样本
x_pred = x_t + σ_t² * score
```

### 采样算法

使用 Ancestral Sampling（祖先采样）：

```python
for t in range(T, 0, -1):
    σ_t = sigma_schedule[t]
    score = score_network(x_t, σ_t)
    x_t = x_t + (σ_t - σ_{t-1}) * score + noise
```

---

## 🎯 下一步

1. ✅ 在 MNIST 上验证基础功能
2. ⏳ 扩展到 CIFAR-10
3. ⏳ 在 SDXL latent space 上测试
4. ⏳ 与原始 SDXL 进行定量对比
5. ⏳ 优化采样速度

---

## 📞 支持

如有问题，请参考：
- 详细架构说明：`architecture_redesign.md`
- 项目文档：`CLAUDE.md`
- 原始论文和代码：`train_combined.py`

**重要提示**：此实现完全独立，不会修改任何现有文件，可以与原有的 Stable Diffusion 代码共存！
