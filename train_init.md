# Stability AI 扩散模型训练流程完整解析

## 目录
1. [训练流程概览](#1-训练流程概览)
2. [数学基础与公式](#2-数学基础与公式)
3. [完整训练算法](#3-完整训练算法)
4. [代码实现详解](#4-代码实现详解)
5. [可替换点分析](#5-可替换点分析)

---

## 1. 训练流程概览

### 1.1 整体架构

```
数据加载
    ↓
VAE 编码 (图像 → 潜在空间)
    ↓
添加噪声 (前向扩散)
    ↓
模型预测 (UNet + Denoiser)
    ↓
计算损失 (与目标对比)
    ↓
反向传播
    ↓
更新权重
```

### 1.2 训练循环

**代码位置：** `sgm/models/diffusion.py:165-187`

```python
def training_step(self, batch, batch_idx):
    # 1. 获取输入并编码到潜在空间
    loss, loss_dict = self.shared_step(batch)

    # 2. 记录日志
    self.log_dict(loss_dict, ...)

    # 3. 返回损失进行反向传播
    return loss
```

---

## 2. 数学基础与公式

### 2.1 前向扩散过程（固定）

**定义：**
$$\mathbf{x}_t = \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

其中：
- $\mathbf{x}_0$：干净数据（目标）
- $\mathbf{x}_t$：噪声数据（输入）
- $\sigma_t$：噪声级别
- $\boldsymbol{\epsilon}$：标准高斯噪声

**代码位置：** `sgm/modules/diffusionmodules/loss.py:42-46, 84`

```python
noise = torch.randn_like(input)              # ε ~ N(0, I)
sigmas_bc = append_dims(sigmas, input.ndim)  # σ_t
noised_input = input + noise * sigmas_bc     # x_t = x_0 + σ_t * ε
```

### 2.2 模型输出（当前实现）

**去噪样本预测：**
$$\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t, \mathbf{c}) = D_\theta(\mathbf{x}_t, \sigma_t, \mathbf{c})$$

其中：
- $D_\theta$：去噪器（Denoiser）
- $\mathbf{c}$：条件（文本、类别等）
- $\hat{\mathbf{x}}_\theta$：预测的干净数据

**实际实现（带预条件）：**
$$\boxed{\hat{\mathbf{x}}_\theta = c_\text{skip}(\sigma_t) \cdot \mathbf{x}_t + c_\text{out}(\sigma_t) \cdot F_\theta(c_\text{in}(\sigma_t) \cdot \mathbf{x}_t, c_\text{noise}(\sigma_t), \mathbf{c})}$$

其中：
- $F_\theta$：原始网络（UNet）
- $c_\text{skip}, c_\text{out}, c_\text{in}, c_\text{noise}$：预条件系数（依赖 $\sigma_t$）

**代码位置：** `sgm/modules/diffusionmodules/denoiser.py:23-39`

```python
def forward(self, network, input, sigma, cond, **kwargs):
    sigma = append_dims(sigma, input.ndim)
    c_skip, c_out, c_in, c_noise = self.scaling(sigma)
    return (
        network(input * c_in, c_noise, cond, **kwargs) * c_out
        + input * c_skip
    )
```

### 2.3 训练目标（损失函数）

**加权 MSE 损失：**
$$\boxed{\mathcal{L}(\theta) = \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}, \sigma_t} \left[ w(\sigma_t) \left\| \hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t, \mathbf{c}) - \mathbf{x}_0 \right\|_2^2 \right]}$$

其中：
- $w(\sigma_t)$：噪声级别相关的权重函数
- $\|\cdot\|_2^2$：L2 范数平方（MSE）

**代码位置：** `sgm/modules/diffusionmodules/loss.py:59-90, 92-96`

```python
# 采样噪声级别
sigmas = self.sigma_sampler(input.shape[0])

# 添加噪声
noised_input = input + noise * sigmas_bc

# 模型预测
model_output = denoiser(network, noised_input, sigmas, cond)

# 计算权重
w = self.loss_weighting(sigmas)

# 损失
loss = torch.mean(w * (model_output - input) ** 2)
```

### 2.4 不同的预条件方案

#### A. EpsScaling（DDPM 风格，预测噪声）

**预条件系数：**
$$\begin{aligned}
c_\text{skip} &= 1 \\
c_\text{out} &= -\sigma_t \\
c_\text{in} &= \frac{1}{\sqrt{\sigma_t^2 + 1}} \\
c_\text{noise} &= \sigma_t
\end{aligned}$$

**去噪样本公式：**
$$\hat{\mathbf{x}}_\theta = \mathbf{x}_t - \sigma_t \cdot F_\theta\left(\frac{\mathbf{x}_t}{\sqrt{\sigma_t^2 + 1}}, \sigma_t, \mathbf{c}\right)$$

**解释：** $F_\theta$ 预测噪声 $\boldsymbol{\epsilon}$

**代码位置：** `sgm/modules/diffusionmodules/denoiser_scaling.py:29-37`

#### B. VScaling（速度预测）

**预条件系数：**
$$\begin{aligned}
c_\text{skip} &= \frac{1}{\sigma_t^2 + 1} \\
c_\text{out} &= \frac{-\sigma_t}{\sqrt{\sigma_t^2 + 1}} \\
c_\text{in} &= \frac{1}{\sqrt{\sigma_t^2 + 1}} \\
c_\text{noise} &= \sigma_t
\end{aligned}$$

**去噪样本公式：**
$$\hat{\mathbf{x}}_\theta = \frac{1}{\sigma_t^2 + 1} \mathbf{x}_t - \frac{\sigma_t}{\sqrt{\sigma_t^2 + 1}} F_\theta\left(\frac{\mathbf{x}_t}{\sqrt{\sigma_t^2 + 1}}, \sigma_t, \mathbf{c}\right)$$

**解释：** $F_\theta$ 预测速度（数据和噪声的组合）

**代码位置：** `sgm/modules/diffusionmodules/denoiser_scaling.py:40-48`

#### C. EDMScaling（Karras 等，2022）

**预条件系数：**
$$\begin{aligned}
c_\text{skip} &= \frac{\sigma_\text{data}^2}{\sigma_t^2 + \sigma_\text{data}^2} \\
c_\text{out} &= \frac{\sigma_t \sigma_\text{data}}{\sqrt{\sigma_t^2 + \sigma_\text{data}^2}} \\
c_\text{in} &= \frac{1}{\sqrt{\sigma_t^2 + \sigma_\text{data}^2}} \\
c_\text{noise} &= \frac{1}{4} \ln \sigma_t
\end{aligned}$$

其中 $\sigma_\text{data}$ 通常取 0.5。

**去噪样本公式：**
$$\hat{\mathbf{x}}_\theta = \frac{\sigma_\text{data}^2}{\sigma_t^2 + \sigma_\text{data}^2} \mathbf{x}_t + \frac{\sigma_t \sigma_\text{data}}{\sqrt{\sigma_t^2 + \sigma_\text{data}^2}} F_\theta\left(\frac{\mathbf{x}_t}{\sqrt{\sigma_t^2 + \sigma_\text{data}^2}}, \frac{\ln \sigma_t}{4}, \mathbf{c}\right)$$

**解释：** $F_\theta$ 预测缩放残差

**代码位置：** `sgm/modules/diffusionmodules/denoiser_scaling.py:15-26`

### 2.5 噪声级别采样

**EDM 采样（对数正态分布）：**
$$\log \sigma_t \sim \mathcal{N}(p_\text{mean}, p_\text{std}^2)$$

$$\sigma_t = \exp(\log \sigma_t)$$

**默认参数：** $p_\text{mean} = -1.2$，$p_\text{std} = 1.2$

**代码位置：** `sgm/modules/diffusionmodules/sigma_sampling.py:6-13`

```python
class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples, rand=None):
        log_sigma = self.p_mean + self.p_std * torch.randn((n_samples,))
        return log_sigma.exp()
```

### 2.6 损失权重

#### A. EDM 权重

$$w(\sigma_t) = \frac{\sigma_t^2 + \sigma_\text{data}^2}{(\sigma_t \cdot \sigma_\text{data})^2}$$

**代码位置：** `sgm/modules/diffusionmodules/denoiser_weighting.py:9-14`

#### B. Eps 权重

$$w(\sigma_t) = \sigma_t^{-2}$$

**代码位置：** `sgm/modules/diffusionmodules/denoiser_weighting.py:22-24`

#### C. Unit 权重

$$w(\sigma_t) = 1$$

**代码位置：** `sgm/modules/diffusionmodules/denoiser_weighting.py:4-6`

---

## 3. 完整训练算法

### 3.1 伪代码

```
算法：扩散模型训练（Stability AI 实现）

输入：
    - 训练数据集 D = {x_0^(i)}
    - 模型 F_θ（UNet）
    - 去噪器缩放 scaling(σ)
    - 条件器 conditioner
    - 噪声采样器 sigma_sampler
    - 损失权重 loss_weighting
    - 第一阶段模型 VAE（可选）

输出：训练好的参数 θ

1. 初始化网络参数 θ
2. for epoch = 1 to num_epochs:
3.     for each batch {x_0^(i), c^(i)} in D:
4.         # ========== 数据准备 ==========
5.         if 使用 VAE:
6.             z_0 = VAE.encode(x_0)       # 编码到潜在空间
7.             z_0 = scale_factor * z_0     # 缩放
8.         else:
9.             z_0 = x_0

10.        # ========== 前向扩散 ==========
11.        σ ~ p(σ)                         # 采样噪声级别（EDM 或离散）
12.        ε ~ N(0, I)                      # 采样标准高斯噪声
13.        z_t = z_0 + σ * ε                # 添加噪声

14.        # ========== 条件编码 ==========
15.        cond = conditioner(c)            # 编码条件（文本、类别等）

16.        # ========== 模型预测（去噪器）==========
17.        c_skip, c_out, c_in, c_noise = scaling(σ)  # 计算预条件系数

18.        # 原始网络输出
19.        F_out = F_θ(z_t * c_in, c_noise, cond)

20.        # 去噪样本预测
21.        ẑ_θ = c_skip * z_t + c_out * F_out

22.        # ========== 计算损失 ==========
23.        w = loss_weighting(σ)            # 计算权重
24.        loss = w * ||ẑ_θ - z_0||²        # 加权 MSE

25.        # ========== 反向传播 ==========
26.        θ ← θ - lr * ∇_θ loss            # 更新参��

27. return θ
```

### 3.2 数学流程（完整公式）

**Step 1: 数据准备**
$$\mathbf{z}_0 = s \cdot \text{VAE}_\text{enc}(\mathbf{x}_0)$$

**Step 2: 噪声级别采样**
$$\sigma_t \sim \mathcal{LN}(p_\text{mean}, p_\text{std}^2)$$

**Step 3: 前向扩散**
$$\mathbf{z}_t = \mathbf{z}_0 + \sigma_t \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

**Step 4: 预条件系数计算**
$$c_\text{skip}(\sigma_t), c_\text{out}(\sigma_t), c_\text{in}(\sigma_t), c_\text{noise}(\sigma_t) = \text{Scaling}(\sigma_t)$$

**Step 5: ��络前向传播**
$$F_\theta = \text{UNet}(c_\text{in}(\sigma_t) \cdot \mathbf{z}_t, c_\text{noise}(\sigma_t), \mathbf{c})$$

**Step 6: 去噪样本预测**
$$\boxed{\hat{\mathbf{z}}_\theta = c_\text{skip}(\sigma_t) \cdot \mathbf{z}_t + c_\text{out}(\sigma_t) \cdot F_\theta}$$

**Step 7: 损失计算**
$$\mathcal{L}(\theta) = w(\sigma_t) \left\| \hat{\mathbf{z}}_\theta - \mathbf{z}_0 \right\|_2^2$$

**Step 8: 梯度下降**
$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)$$

---

## 4. 代码实现详解

### 4.1 完整训练流程代码追踪

#### Level 1: 训���入口

**文件：** `sgm/models/diffusion.py:165-187`

```python
def training_step(self, batch, batch_idx):
    """PyTorch Lightning 训练步骤"""
    # 调用 shared_step
    loss, loss_dict = self.shared_step(batch)

    # 记录日志
    self.log_dict(loss_dict, prog_bar=True, logger=True, ...)

    return loss  # 返回损失用于自动反向传播
```

#### Level 2: 共享步骤（数据准备）

**文件：** `sgm/models/diffusion.py:158-163`

```python
def shared_step(self, batch: Dict) -> Any:
    """训练和验证的共享步骤"""
    # 1. 获取图像数据
    x = self.get_input(batch)           # batch["jpg"]

    # 2. 编码到潜在空间（如果使用 VAE）
    x = self.encode_first_stage(x)      # VAE 编码 + 缩放

    # 3. 添加全局步数到 batch
    batch["global_step"] = self.global_step

    # 4. 调用前向传播（计算损失）
    loss, loss_dict = self(x, batch)    # 调用 forward

    return loss, loss_dict
```

**VAE 编码：** `sgm/models/diffusion.py:138-150`

```python
def encode_first_stage(self, x):
    """编码图像到潜在空间"""
    with torch.autocast("cuda", enabled=not self.disable_first_stage_autocast):
        z = self.first_stage_model.encode(x)  # VAE 编码

    z = self.scale_factor * z                  # 缩放（通常 0.13025）
    return z
```

#### Level 3: 前向传播（损失计算）

**文件：** `sgm/models/diffusion.py:152-156`

```python
def forward(self, x, batch):
    """计算损失"""
    # 调用损失函数
    loss = self.loss_fn(
        self.model,        # 原始网络（UNet，被 wrapper 包装）
        self.denoiser,     # 去噪器（带预条件）
        self.conditioner,  # 条件器
        x,                 # 潜在空间数据 z_0
        batch              # 包含条件信息的 batch
    )

    loss_mean = loss.mean()
    loss_dict = {"loss": loss_mean}
    return loss_mean, loss_dict
```

#### Level 4: 损失函数（核心训练逻辑）

**文件：** `sgm/modules/diffusionmodules/loss.py:48-57`

```python
def forward(
    self,
    network: nn.Module,        # UNet（被 OpenAIWrapper 包装）
    denoiser: Denoiser,        # 去噪器
    conditioner: GeneralConditioner,  # 条件器
    input: torch.Tensor,       # z_0（潜在空间，干净数据）
    batch: Dict,               # 包含条件的 batch
) -> torch.Tensor:
    # 1. 编码条件
    cond = conditioner(batch)

    # 2. 调用内部前向传播
    return self._forward(network, denoiser, cond, input, batch)
```

#### Level 5: 损失函数内部实现（关键！）

**文件：** `sgm/modules/diffusionmodules/loss.py:59-90`

```python
def _forward(
    self,
    network: nn.Module,
    denoiser: Denoiser,
    cond: Dict,
    input: torch.Tensor,       # z_0（目标）
    batch: Dict,
) -> Tuple[torch.Tensor, Dict]:
    # ========== 1. 采样噪声级别 ==========
    sigmas = self.sigma_sampler(input.shape[0]).to(input)
    # sigmas: [B]，每个样本一个 σ_t

    # ========== 2. 生成噪声 ==========
    noise = torch.randn_like(input)  # ε ~ N(0, I)

    # （可选）Offset 噪声（提高多样性）
    if self.offset_noise_level > 0.0:
        noise = noise + self.offset_noise_level * ...

    # ========== 3. 添加噪声（前向扩散）==========
    sigmas_bc = append_dims(sigmas, input.ndim)  # [B, 1, 1, 1]
    noised_input = input + noise * sigmas_bc     # z_t = z_0 + σ_t * ε

    # ========== 4. 模型预测（去噪器）==========
    model_output = denoiser(
        network,           # UNet
        noised_input,      # z_t
        sigmas,            # σ_t
        cond,              # 条件
        **additional_model_inputs
    )
    # model_output: ẑ_θ（预测的去噪样本）

    # ========== 5. 计算损失 ==========
    w = append_dims(self.loss_weighting(sigmas), input.ndim)
    # w: 权重，形状 [B, 1, 1, 1]

    return self.get_loss(model_output, input, w)
    # 返回：w * ||model_output - input||²
```

#### Level 6: 去噪器（预条件）

**文件：** `sgm/modules/diffusionmodules/denoiser.py:23-39`

```python
def forward(
    self,
    network: nn.Module,    # UNet
    input: torch.Tensor,   # z_t（噪声输入）
    sigma: torch.Tensor,   # σ_t（噪声级别）
    cond: Dict,            # 条件
    **additional_model_inputs,
) -> torch.Tensor:
    # ========== 1. 扩展 sigma 维度 ==========
    sigma = self.possibly_quantize_sigma(sigma)  # [B]
    sigma_shape = sigma.shape
    sigma = append_dims(sigma, input.ndim)       # [B, 1, 1, 1]

    # ========== 2. 计算预条件系数 ==========
    c_skip, c_out, c_in, c_noise = self.scaling(sigma)
    # 根据不同的 scaling 方案（EpsScaling, VScaling, EDMScaling）

    # ========== 3. 调用网络 ==========
    c_noise = self.possibly_quantize_c_noise(c_noise.reshape(sigma_shape))
    network_output = network(
        input * c_in,      # 预条件输入
        c_noise,           # 噪声级别（传给网络的 timesteps）
        cond,              # 条件
        **additional_model_inputs
    )

    # ========== 4. 应用预条件得到去噪样本 ==========
    return network_output * c_out + input * c_skip
    # ẑ_θ = c_out * F_θ + c_skip * z_t
```

#### Level 7: 网络（UNet）

**文件：** `sgm/modules/diffusionmodules/openaimodel.py:866-903`

```python
def forward(
    self,
    x: torch.Tensor,         # input * c_in
    timesteps: torch.Tensor, # c_noise（噪声级别）
    context: torch.Tensor = None,  # 交叉注意力条件
    y: torch.Tensor = None,        # 向量条件
    **kwargs,
) -> torch.Tensor:
    """UNet 前向传播"""
    # 1. 时间嵌入
    t_emb = timestep_embedding(timesteps, self.model_channels)
    emb = self.time_embed(t_emb)

    # 2. 类别嵌入（如果有）
    if self.num_classes is not None:
        emb = emb + self.label_emb(y)

    # 3. 编码器
    h = x
    hs = []
    for module in self.input_blocks:
        h = module(h, emb, context)
        hs.append(h)

    # 4. 中间层
    h = self.middle_block(h, emb, context)

    # 5. 解码器
    for module in self.output_blocks:
        h = torch.cat([h, hs.pop()], dim=1)  # 跳跃连接
        h = module(h, emb, context)

    # 6. 输出
    return self.out(h)  # F_θ
```

#### Level 8: 损失计算

**文件：** `sgm/modules/diffusionmodules/loss.py:92-96`

```python
def get_loss(self, model_output, target, w):
    """计算加权 MSE 损失"""
    if self.loss_type == "l2":
        return torch.mean(
            (w * (model_output - target) ** 2).reshape(target.shape[0], -1),
            dim=1
        )
    elif self.loss_type == "l1":
        return torch.mean(
            (w * (model_output - target).abs()).reshape(target.shape[0], -1),
            dim=1
        )
    # ...
```

### 4.2 数据流图解

```
┌──────────────────────────────��──────────────────────────────┐
│                      训练数据 batch                          │
│  {"jpg": images, "txt": texts, ...}                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  shared_step (diffusion.py:158)                             │
│  ┌───────────────────────────────────────┐                 │
│  │ 1. x = batch["jpg"]                    │                 │
│  │ 2. z_0 = VAE.encode(x) * scale_factor  │                 │
│  └───────────────────────────────────────┘                 │
└────────────────────┬────────────────────────────────────────┘
                     │ z_0（潜在空间，干净数据）
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  forward (diffusion.py:152)                                 │
│  ┌───────────────────────────────────────┐                 │
│  │ loss = loss_fn(model, denoiser,       │                 │
│  │               conditioner, z_0, batch) │                 │
│  └───────────────────────────────────────┘                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  StandardDiffusionLoss._forward (loss.py:59)                │
│  ┌───────────────────────────────────────┐                 │
│  │ 1. σ ~ p(σ)              [采样]       │                 │
│  │ 2. ε ~ N(0, I)           [采样]       │                 │
│  │ 3. z_t = z_0 + σ * ε     [加噪]       │                 │
│  │ 4. cond = conditioner(batch) [条件]   │                 │
│  │ 5. ẑ_θ = denoiser(...)   [预测]       │ ← 关键！
│  │ 6. loss = w * ||ẑ_θ - z_0||² [损失]   │                 │
│  └───────────────────────────────────────┘                 │
└────────────────────┬────────────────────────────────────────┘
                     │ 进入 denoiser
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Denoiser.forward (denoiser.py:23)                          │
│  ┌───────────────────────────────────────┐                 │
│  │ 1. c_skip, c_out, c_in, c_noise       │                 │
│  │      = scaling(σ)        [预条件]     │                 │
│  │ 2. F_θ = network(z_t * c_in,          │                 │
│  │                  c_noise, cond)        │ ← UNet 输出
│  │ 3. ẑ_θ = c_skip * z_t + c_out * F_θ   │ ← 去噪样本
│  └───────────────────────────────────────┘                 │
└────────────────────┬────────────────────────────────────────┘
                     │ F_θ（网络输出）
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  UNetModel.forward (openaimodel.py:866)                     │
│  ┌───────────────────────────────────────┐                 │
│  │ 输入：z_t * c_in, c_noise, cond       │                 │
│  │ 输出：F_θ（残差/噪声/速度预测）      │                 │
│  └───────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 可替换点分析

### 5.1 关键替换位置

#### ⭐ **替换点 1：Denoiser 输出（推荐）**

**位置：** `sgm/modules/diffusionmodules/denoiser.py:36-39`

**当前实现：**
```python
return (
    network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out
    + input * c_skip
)
# 输出：去噪样本 ẑ_θ
```

**替换为分数函数：**

如果你的模型输出分数 $\mathbf{s}_\theta$，在这里添加转换：

```python
# 当前代码（去噪样本）
# ẑ_θ = c_skip * z_t + c_out * F_θ

# 如果 F_θ 输出分数，需要转换
network_output = network(input * c_in, c_noise, cond, **additional_model_inputs)

# 判断是否是分数输出
if self.is_score_output:  # 新增标志
    # 转换：ẑ_θ = z_t + σ² * score
    score = network_output
    sigma = append_dims(sigma, input.ndim)  # 确保 sigma 已扩展
    denoised = input + (sigma ** 2) * score
    return denoised
else:
    # 原始逻辑
    return network_output * c_out + input * c_skip
```

**数学公式：**

**原始（去噪样本）：**
$$\hat{\mathbf{z}}_\theta = c_\text{skip}(\sigma_t) \cdot \mathbf{z}_t + c_\text{out}(\sigma_t) \cdot F_\theta$$

**替换（分数函数）：**
$$\boxed{\hat{\mathbf{z}}_\theta = \mathbf{z}_t + \sigma_t^2 \cdot \mathbf{s}_\theta}$$

其中 $\mathbf{s}_\theta = F_\theta(\mathbf{z}_t, \sigma_t, \mathbf{c})$ 是你的模型输出。

---

#### ⭐ **替换点 2：自定义 Scaling 类**

**位置：** `sgm/modules/diffusionmodules/denoiser_scaling.py`

**创建新的 ScoreScaling 类：**

```python
class ScoreScaling:
    """
    用于输出分数函数的模型

    网络输出：score = ∇log p(z_t)
    去噪样本：ẑ_θ = z_t + σ² * score

    去噪器公式：ẑ_θ = c_skip * z_t + c_out * network_output
    因此：c_skip = 1, c_out = σ²
    """
    def __call__(
        self, sigma: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c_skip = torch.ones_like(sigma, device=sigma.device)  # 1
        c_out = sigma ** 2                                     # σ²
        c_in = torch.ones_like(sigma, device=sigma.device)    # 1
        c_noise = sigma.clone()                                # σ
        return c_skip, c_out, c_in, c_noise
```

**配置文件使用：**
```yaml
denoiser_config:
  target: sgm.modules.diffusionmodules.denoiser.Denoiser
  params:
    scaling_config:
      target: sgm.modules.diffusionmodules.denoiser_scaling.ScoreScaling
```

**数学验证：**
$$\begin{aligned}
\hat{\mathbf{z}}_\theta &= c_\text{skip} \cdot \mathbf{z}_t + c_\text{out} \cdot \mathbf{s}_\theta \\
&= 1 \cdot \mathbf{z}_t + \sigma_t^2 \cdot \mathbf{s}_\theta \\
&= \mathbf{z}_t + \sigma_t^2 \cdot \mathbf{s}_\theta \quad \checkmark
\end{aligned}$$

---

#### ⭐ **替换点 3：Network 包装器**

**位置：** `sgm/modules/diffusionmodules/wrappers.py` 或创建新文件

**创建 ScoreToDenoised 包装器：**

```python
class ScoreToDenoised(nn.Module):
    """将分数模型包装为去噪模型"""

    def __init__(self, score_model: nn.Module):
        super().__init__()
        self.score_model = score_model

    def forward(
        self,
        x: torch.Tensor,        # z_t * c_in
        timesteps: torch.Tensor,  # c_noise（σ 或其变换）
        context: torch.Tensor = None,
        y: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        # 1. 调用分数模型
        score = self.score_model(x, timesteps, context, y, **kwargs)

        # 2. 这里不需要转换！
        # 因为 denoiser 会处理：ẑ = c_skip * z_t + c_out * score
        # 只要配置正确的 scaling（c_out = σ²），就能得到正确结果

        return score  # 直接返回分数
```

**配置使用：**
```yaml
network_config:
  target: sgm.modules.diffusionmodules.wrappers.OpenAIWrapper
  params:
    diffusion_model:
      target: path.to.ScoreToDenoised
      params:
        score_model:
          target: your_module.YourScoreModel
          params:
            # 你的模型参数
```

---

### 5.2 三种替换方案对比

| 方案 | 修改位置 | 复杂度 | 灵活性 | 推荐度 |
|------|---------|--------|--------|--------|
| **方案 1：修改 Denoiser** | `denoiser.py:36-39` | 低 | 中 | ⭐⭐⭐ |
| **方案 2：自定义 Scaling** | `denoiser_scaling.py` | 低 | 高 | ⭐⭐⭐⭐⭐ |
| **方案 3：Network 包装器** | 创建新文件 | 中 | 高 | ⭐⭐⭐⭐ |

**推荐：方案 2（自定义 Scaling）** - 最干净，最符合框架设计

---

### 5.3 完整替换示例

#### 假设你的模型

```python
class YourScoreModel(nn.Module):
    """你的分数函数模型"""

    def forward(
        self,
        x: torch.Tensor,           # z_t（或 z_t * c_in）
        timesteps: torch.Tensor,   # σ_t（或其变换 c_noise）
        context: torch.Tensor = None,
        y: torch.Tensor = None,
        **kwargs
    ) -> torch.Tensor:
        """
        输出：分数函数 ∇log p(z_t)
        """
        # 你的实现
        # ...
        return score  # [B, C, H, W]
```

#### 使用方案 2（自定义 Scaling）

**Step 1: 创建 ScoreScaling**

在 `sgm/modules/diffusionmodules/denoiser_scaling.py` 添加：

```python
class ScoreScaling:
    def __call__(self, sigma: torch.Tensor):
        c_skip = torch.ones_like(sigma)
        c_out = sigma ** 2
        c_in = torch.ones_like(sigma)
        c_noise = sigma.clone()
        return c_skip, c_out, c_in, c_noise
```

**Step 2: 配置文件**

```yaml
model:
  target: sgm.models.diffusion.DiffusionEngine
  params:
    network_config:
      target: sgm.modules.diffusionmodules.wrappers.OpenAIWrapper
      params:
        diffusion_model:
          target: your_module.YourScoreModel
          params:
            # 你的参数

    denoiser_config:
      target: sgm.modules.diffusionmodules.denoiser.Denoiser
      params:
        scaling_config:
          target: sgm.modules.diffusionmodules.denoiser_scaling.ScoreScaling

    # ... 其他配置
```

**Step 3: 训练**

```bash
python main.py --base configs/your_config.yaml
```

**数据流：**
```
z_t（噪声数据）
    ↓
Denoiser:
    c_skip=1, c_out=σ², c_in=1, c_noise=σ
    ↓
YourScoreModel(z_t * 1, σ, cond) → score
    ↓
ẑ_θ = 1 * z_t + σ² * score = z_t + σ² * score  ✓
    ↓
Loss = w * ||ẑ_θ - z_0||²
```

---

### 5.4 关键点总结

#### ✅ 可以替换的地方

1. **Denoiser.forward**（直接修改返回值）
2. **Denoiser Scaling**（修改预条件系数，推荐）
3. **Network 包装器**（在模型外部添加转换层）
4. **Loss 函数**（如果要改训练目标）

#### ❌ 不需要替换的地方

1. **前向扩散**（`z_t = z_0 + σ * ε`）- 保持不变
2. **Sigma 采样**（EDMSampling 等）- 保持不变
3. **条件器**（Conditioner）- 保持不变
4. **VAE 编码/解码**- 保持不变
5. **损失计算**（MSE）- 保持不变

#### 🔑 核心数学关系

**无论你的模型输出什么，最终都要转换为去噪样本：**

$$\hat{\mathbf{z}}_\theta = \begin{cases}
\mathbf{z}_t - \sigma_t \boldsymbol{\epsilon}_\theta & \text{噪声预测} \\
\mathbf{z}_t + \sigma_t^2 \mathbf{s}_\theta & \text{分数预测} \\
\text{直接输��} & \text{去噪样本预测}
\end{cases}$$

**训练目标始终是：**
$$\min_\theta \mathbb{E} \left[ w(\sigma_t) \left\| \hat{\mathbf{z}}_\theta - \mathbf{z}_0 \right\|^2 \right]$$

---

## 6. 总结

### 6.1 训练流程公式总结

$$\begin{aligned}
\text{数据：} & \quad \mathbf{z}_0 = s \cdot \text{VAE}_\text{enc}(\mathbf{x}_0) \\
\text{采样：} & \quad \sigma_t \sim p(\sigma), \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I}) \\
\text{加噪：} & \quad \mathbf{z}_t = \mathbf{z}_0 + \sigma_t \boldsymbol{\epsilon} \\
\text{条件：} & \quad \mathbf{c} = \text{Conditioner}(\text{batch}) \\
\text{预测：} & \quad \hat{\mathbf{z}}_\theta = c_\text{skip}(\sigma_t) \mathbf{z}_t + c_\text{out}(\sigma_t) F_\theta(c_\text{in}(\sigma_t) \mathbf{z}_t, c_\text{noise}(\sigma_t), \mathbf{c}) \\
\text{损失：} & \quad \mathcal{L} = w(\sigma_t) \|\hat{\mathbf{z}}_\theta - \mathbf{z}_0\|^2
\end{aligned}$$

### 6.2 分数函数替换要点

**如果你的模型输出分数 $\mathbf{s}_\theta$：**

1. **转换公式：** $\hat{\mathbf{z}}_\theta = \mathbf{z}_t + \sigma_t^2 \mathbf{s}_\theta$

2. **实现方式：** 使用 `ScoreScaling`（$c_\text{skip}=1, c_\text{out}=\sigma^2$）

3. **无需修改：** 损失函数、前向扩散、采样器等保持不变

4. **sigma 可获取：** 在所有地方都可以访问 $\sigma_t$

### 6.3 代码关键位置

| 功能 | 文件 | 行号 | 说明 |
|-----|------|-----|------|
| 训练入口 | `diffusion.py` | 165-187 | training_step |
| 数据准备 | `diffusion.py` | 158-163 | VAE 编码 |
| 损失计算 | `loss.py` | 59-90 | 前向扩散 + 预测 |
| 去噪器 | `denoiser.py` | 23-39 | 预条件 + 网络调用 |
| 预条件 | `denoiser_scaling.py` | 全文 | c_skip, c_out, c_in, c_noise |
| UNet | `openaimodel.py` | 866-903 | 网络前向传播 |

**这就是完整的训练流程分析！**
