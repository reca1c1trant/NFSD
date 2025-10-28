# Stable Diffusion 架构重设计：从 UNet 到 Normalizing Flow

## 一、原有的 Stable Diffusion 架构

### 1.1 核心组件

在当前的 `generative-models` 项目中，Stable Diffusion 的架构包含以下关键组件：

#### **DiffusionEngine** (`sgm/models/diffusion.py`)
主要的扩散模型类，协调所有组件工作。

#### **UNet Network** (`sgm/modules/diffusionmodules/openaimodel.py`)
- **作用**：神经网络主干，接收带噪声的图像 $\mathbf{x}_t$ 和时间步 $t$
- **输出**：预测值 $\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t)$（可以是预测的原始图像，也可以是噪声）
- **结构**：深度卷积网络，包含下采样、上采样、残差块、注意力机制
- **参数量**：数十亿参数（如 SDXL 有约 2.6B 参数）

#### **Denoiser** (`sgm/modules/diffusionmodules/denoiser.py`)
- **作用**：实现去噪框架，将 UNet 的输出转换为去噪样本
- **包含子组件**：
  - **Weighting** (`denoiser_weighting.py`)：损失函数权重
  - **Scaling** (`denoiser_scaling.py`)：输入输出的预处理
  - **Sigma Sampling** (`sigma_sampling.py`)：噪声水平采样策略

#### **Sampler** (`sgm/modules/diffusionmodules/sampling.py`)
- **作用**：数值求解器，如 Euler、DDPM、DDIM
- **独立于模型**：不需要修改网络就能切换采样器

### 1.2 原有的工作流程

```
训练阶段：
1. 从真实数据 x_0 开始
2. 加噪声得到 x_t = x_0 + σ_t * ε（ε ~ N(0,I)）
3. UNet 预测：x̂_θ = UNet(x_t, σ_t, conditions)
4. 计算损失：L = ||x̂_θ - x_0||² （或其他变体）
5. 反向传播更新 UNet 参数

采样阶段（去噪）：
1. 从纯噪声 x_T ~ N(0, σ_max² I) 开始
2. 迭代 t = T, T-1, ..., 1：
   a. UNet 预测：x̂_θ = UNet(x_t, σ_t, conditions)
   b. 计算 score function：∇log p_t(x_t) = (x̂_θ - x_t) / σ_t²
   c. 使用 sampler 更新：x_{t-1} = Sampler(x_t, score, σ_t)
3. 得到最终去噪样本 x_0
```

### 1.3 关键公式

在原有架构中，**score function** 的计算是核心：

$$\boxed{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \frac{\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t) - \mathbf{x}_t}{\sigma_t^2}}$$

这个公式的含义：
- **左边**：概率密度对数的梯度（score function），指向概率密度增加最快的方向
- **右边**：由 UNet 的预测值 $\hat{\mathbf{x}}_\theta$ 计算得到

---

## 二、你提出的新架构：Normalizing Flow 替代方案

### 2.1 核心思想

从 `train_combined.py` 中的 `NormalizingFlow` 类出发：

#### **NormalizingFlow 的能力**
```python
class NormalizingFlow(nn.Module):
    def __init__(self, dim, n_flows, activation, sampler):
        # 基础分布：高斯分布
        self.base_dist = dist.MultivariateNormal(torch.zeros(dim), torch.eye(dim))

        # Flow 层：多个可逆变换
        self.flows = nn.ModuleList([FlowLayer(...) for _ in range(n_flows)])

        # 最终线性变换
        self.final_mlp = nn.Linear(dim, dim)

    def forward_with_base_samples(self, z_base):
        # 从高斯分布 -> 目标分布
        # 返回：样本 x 和 log_prob_model（对数概率密度）
        ...
        return x, log_prob_model
```

**关键优势**：
1. ✅ **能计算精确的 log probability**：$\log p_\theta(\mathbf{x})$
2. ✅ **可以求梯度得到 score function**：$\nabla_\mathbf{x} \log p_\theta(\mathbf{x})$
3. ✅ **参数量更小**：只需要几层线性变换 + 激活函数，而不是深度 UNet
4. ✅ **理论基础更清晰**：直接建模概率分布

### 2.2 新架构设计

#### **用 NormalizingFlow 替代 UNet + Denoiser**

```
新的组件结构：

1. ScoreFlowNetwork (替代 UNet)
   - 基于 NormalizingFlow
   - 输入：带噪声的样本 x_t，噪声水平 σ_t，条件信息 c
   - 输出：score function ∇log p_t(x_t)

2. 直接计算去噪样本（不需要单独的 Denoiser）
   - 从 score 直接得到预测：x̂_θ = x_t + σ_t² * score
   - 或者：用 score 进行梯度上升采样

3. Sampler（保持不变）
   - 使用 score 进行迭代采样
```

#### **具体实现路径**

```python
class ScoreFlowNetwork(nn.Module):
    """基于 Normalizing Flow 的 Score Network"""

    def __init__(self, dim, n_flows, activation, conditioner_config):
        super().__init__()

        # Normalizing Flow 核心
        self.flow = NormalizingFlow(dim, n_flows, activation, sampler)

        # 条件编码器（保持原有的 GeneralConditioner）
        self.conditioner = instantiate_from_config(conditioner_config)

        # 噪声水平嵌入
        self.sigma_embedder = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, dim)
        )

    def forward(self, x_t, sigma_t, conditions):
        """
        输入：
            x_t: 带噪声的样本 [B, C, H, W] 或 [B, D]
            sigma_t: 噪声水平 [B]
            conditions: 条件信息字典（text, class, etc.）

        输出：
            score: ∇log p_t(x_t) [B, C, H, W] 或 [B, D]
            x_pred: 预测的干净样本 [B, C, H, W] 或 [B, D]
        """
        batch_size = x_t.shape[0]

        # 1. 编码条件信息
        cond_embeddings = self.conditioner(conditions)  # [B, D_cond]

        # 2. 编码噪声水平
        sigma_embedding = self.sigma_embedder(sigma_t.unsqueeze(-1))  # [B, D]

        # 3. 将 x_t, sigma, conditions 组合
        # 可以通过 concatenation 或者 cross-attention
        flow_input = self._combine_inputs(x_t, sigma_embedding, cond_embeddings)

        # 4. 通过 Normalizing Flow（支持梯度计算）
        x_t.requires_grad_(True)
        _, log_prob = self.flow.forward_with_base_samples(flow_input)

        # 5. 计算 score function = ∇_x log p(x)
        score = torch.autograd.grad(
            outputs=log_prob.sum(),
            inputs=x_t,
            create_graph=True  # 允许二阶导数用于训练
        )[0]

        # 6. 从 score 恢复预测的干净样本
        # 根据公式：∇log p_t(x_t) = (x̂_θ - x_t) / σ_t²
        # 反推：x̂_θ = x_t + σ_t² * score
        x_pred = x_t + (sigma_t ** 2).view(-1, 1) * score

        return score, x_pred

    def _combine_inputs(self, x_t, sigma_emb, cond_emb):
        """组合输入：x_t + sigma + conditions"""
        # 简单版本：concatenation
        # 高级版本：cross-attention, FiLM, AdaGN 等
        return torch.cat([x_t, sigma_emb, cond_emb], dim=-1)
```

### 2.3 训练流程

```python
# 原有的训练循环（in main.py or DiffusionEngine）
def training_step(self, batch):
    # 1. 获取数据
    x_0 = batch['jpg']  # 原始干净图像
    conditions = batch  # 条件信息

    # 2. 采样噪声水平
    sigma_t = self.sigma_sampler.sample(batch_size)

    # 3. 加噪声
    noise = torch.randn_like(x_0)
    x_t = x_0 + sigma_t.view(-1, 1, 1, 1) * noise

    # 4. 模型预测（新架构）
    score, x_pred = self.score_flow_network(x_t, sigma_t, conditions)

    # 5. 计算损失（保持不变）
    # 可以用原有的 StandardDiffusionLoss
    loss = torch.mean((x_pred - x_0) ** 2)  # 简化版

    # 或者用 score matching loss
    # target_score = -(x_t - x_0) / (sigma_t ** 2).view(-1, 1, 1, 1)
    # loss = torch.mean((score - target_score) ** 2)

    return loss
```

### 2.4 采样流程

```python
# 新的采样流程
def sample(self, conditions, num_steps=50):
    batch_size = conditions['txt'].shape[0]

    # 1. 从纯噪声开始
    x_t = torch.randn(batch_size, C, H, W) * sigma_max

    # 2. 迭代去噪
    for t in range(num_steps, 0, -1):
        sigma_t = self.get_sigma(t, num_steps)

        # 3. 计算 score
        with torch.no_grad():
            score, x_pred = self.score_flow_network(x_t, sigma_t, conditions)

        # 4. 更新样本（使用 Euler sampler 为例）
        if t > 1:
            sigma_next = self.get_sigma(t - 1, num_steps)
            x_t = x_t + (sigma_t - sigma_next) * score
        else:
            # 最后一步直接用预测值
            x_t = x_pred

    return x_t
```

---

## 三、新旧架构对比

### 3.1 组件映射

| 原有架构 | 新架构 | 说明 |
|---------|--------|------|
| **UNet** | **NormalizingFlow** | 从深度卷积网络变为可逆变换序列 |
| **Denoiser** | **直接用 score 公式** | 不需要单独的 denoiser 模块 |
| **network_config** | **flow_config** | 配置 Flow 层数、激活函数 |
| **条件编码器** | **保持不变** | 继续使用 GeneralConditioner |
| **Sampler** | **保持不变** | 继续使用 Euler/DDPM/DDIM |
| **Loss** | **保持不变** | 可以用同样的损失函数 |

### 3.2 优势对比

| 方面 | 原有 UNet | 新 NormalizingFlow |
|------|-----------|-------------------|
| **参数量** | 数十亿（~2.6B） | 数百万（取决于层数） |
| **理论基础** | 经验性强 | 精确的概率建模 |
| **计算 score** | 间接（通过预测 x̂） | 直接（梯度计算） |
| **训练稳定性** | 需要大量数据和算力 | Flow 训练相对简单 |
| **可解释性** | 黑盒 | 每一层都是可逆变换 |
| **灵活性** | 很灵活（已验证） | 需要验证在复杂数据上的表现 |

### 3.3 挑战和考虑

#### **潜在问题**：
1. ❓ **维度问题**：图像是高维数据（如 256×256×3 = 196,608 维），Normalizing Flow 在高维空间的表达能力是否足够？
2. ❓ **空间结构**：UNet 利用了图像的局部性和空间结构，Flow 需要如何处理？
3. ❓ **计算效率**：每次前向都需要计算梯度（autograd），可能比 UNet 慢
4. ❓ **条件注入**：如何有效地将文本、类别等条件注入到 Flow 中？

#### **可能的解决方案**：
1. ✅ **先在 latent space 工作**：继续使用 VAE 编码，在低维 latent space 上应用 Flow
2. ✅ **分块处理**：将图像分成 patches，每个 patch 用独立的 Flow
3. ✅ **混合架构**：用轻量级 CNN 提取特征，再用 Flow 建模分布
4. ✅ **条件 Flow**：参考 Conditional Flow Matching 的方法

---

## 四、实施路径建议

### 阶段 1：概念验证（玩具数据）
1. 在低维数据（如 MNIST latent，维度 ~64）上测试
2. 实现 `ScoreFlowNetwork` 的基础版本
3. 验证 score 计算的正确性
4. 对比与原 UNet 的性能

### 阶段 2：扩展到 SDXL latent space
1. 在 SDXL 的 VAE latent space（4×64×64 = 16,384 维）上测试
2. 优化 Flow 的层数和激活函数
3. 集成条件编码器（text encoder, etc.）
4. 测试采样质量

### 阶段 3：完整替换
1. 修改 `sgm/models/diffusion.py` 中的 `DiffusionEngine`
2. 创建新的配置文件 `configs/training/flow_diffusion.yaml`
3. 完整的训练和评估流程
4. 与原 SDXL 进行定量和定性对比

---

## 五、代码改动清单

### 需要新建的文件：
1. `sgm/models/score_flow.py` - ScoreFlowNetwork 实现
2. `sgm/modules/flows/normalizing_flow.py` - 从 train_combined.py 迁移 Flow 组件
3. `sgm/modules/flows/flow_layers.py` - FlowLayer 实现
4. `configs/training/flow_diffusion_mnist.yaml` - 玩具数据配置
5. `configs/training/flow_diffusion_sdxl.yaml` - SDXL 配置

### 需要修改的文件：
1. `sgm/models/diffusion.py` - DiffusionEngine 支持新的 network_config
2. `main.py` - 添加对 Flow 的支持（可能不需要改动）
3. `sgm/util.py` - 确保 instantiate_from_config 能处理新模块

### 可以保持不变的文件：
1. `sgm/modules/encoders/modules.py` - 条件编码器
2. `sgm/modules/diffusionmodules/sampling.py` - 采样器
3. `sgm/modules/diffusionmodules/loss.py` - 损失函数（可能需要小改动）
4. 所有数据加载相关的代码

---

## 六、总结

### 核心理念
用 **Normalizing Flow 的精确概率建模能力** 替代 **UNet 的强大但黑盒的表达能力**，通过：

$$\text{score} = \nabla_{\mathbf{x}} \log p_\theta(\mathbf{x}) \quad \Rightarrow \quad \hat{\mathbf{x}}_\theta = \mathbf{x}_t + \sigma_t^2 \cdot \text{score}$$

直接获得去噪样本，简化架构，提高可解释性。

### 关键优势
- ✅ 参数量大幅减少
- ✅ 理论基础更清晰
- ✅ 精确的概率建模
- ✅ 可以计算精确的 likelihood

### 主要挑战
- ❓ 高维数据的表达能力
- ❓ 空间结构的建模
- ❓ 计算效率（梯度计算）
- ❓ 与大规模训练的兼容性

这是一个非常有前景的研究方向，值得深入探索！
