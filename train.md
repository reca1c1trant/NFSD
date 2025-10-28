# Stability AI 训练代码流程

## 1. 启动入口

**文件：`main.py`**

```bash
python main.py --base configs/example_training/toy/mnist_diffusion.yaml
```

**关键逻辑：**
- 行 662：`model = instantiate_from_config(config.model)` - 实例化模型
- 行 830：`data = instantiate_from_config(config.data)` - 实例化数据
- 行 825：`trainer = Trainer(**trainer_opt, **trainer_kwargs)` - 创建 PyTorch Lightning Trainer
- 行 900：`trainer.fit(model, data)` - 开始训练

---

## 2. 模型架构

**文件：`sgm/models/diffusion.py`**

### 2.1 DiffusionEngine 类（主模型，继承 pl.LightningModule）

**初始化 (行 19-83)：**
```python
class DiffusionEngine(pl.LightningModule):
    def __init__(
        self,
        network_config,        # UNet 配置
        denoiser_config,       # 去噪器配置
        first_stage_config,    # VAE 配置
        conditioner_config,    # 条件器配置
        loss_fn_config,        # 损失函数配置
        scheduler_config,      # 学习率调度器配置
        ...
    ):
        # 核心组件
        self.model = OpenAIWrapper(UNet)           # 行 48-50: UNet 网络
        self.denoiser = Denoiser(scaling)          # 行 53: 去噪器（带预条件）
        self.conditioner = GeneralConditioner()    # 行 59-61: 条件编码器
        self.first_stage_model = VAE()             # 行 63: VAE（第一阶段模型）
        self.loss_fn = StandardDiffusionLoss()     # 行 65-69: 损失函数
```

### 2.2 训练步骤 (行 165-187)

```python
def training_step(self, batch, batch_idx):
    loss, loss_dict = self.shared_step(batch)  # 调用共享步骤
    self.log_dict(loss_dict, ...)              # 记录日志
    return loss                                 # 返回损失（自动反向传播）
```

### 2.3 共享步骤 (行 158-163)

```python
def shared_step(self, batch: Dict):
    x = self.get_input(batch)           # 获取图像
    x = self.encode_first_stage(x)      # VAE 编码到潜在空间
    loss, loss_dict = self(x, batch)    # 前向传播计算损失
    return loss, loss_dict
```

### 2.4 前向传播 (行 152-156)

```python
def forward(self, x, batch):
    loss = self.loss_fn(
        self.model,        # UNet
        self.denoiser,     # 去噪器
        self.conditioner,  # 条件器
        x,                 # 潜在空间数据 z_0
        batch              # 条件信息
    )
    return loss.mean(), {"loss": loss.mean()}
```

---

## 3. 各组件详解

### 3.1 VAE（第一阶段模型）

**文件：`sgm/models/autoencoder.py`**

**作用：** 编码图像到潜在空间，解码潜在向量到图像

**位置：** `diffusion.py:138-150`
```python
def encode_first_stage(self, x):
    z = self.first_stage_model.encode(x)  # VAE 编码
    z = self.scale_factor * z              # 缩放（通常 0.13025）
    return z
```

**数学：** $z_0 = s \cdot \text{VAE}_\text{enc}(x_0)$

---

### 3.2 Conditioner（条件器）

**文件：`sgm/modules/encoders/modules.py`**

**作用：** 编码文本、类别等条件信息

**关键类：**
- `FrozenCLIPEmbedder` - CLIP 文本编码器
- `FrozenOpenCLIPEmbedder` - OpenCLIP 文本编码器
- `GeneralConditioner` - 通用条件器（组合多个编码器）

**使用位置：** `loss.py:56`
```python
cond = conditioner(batch)  # 编码条件
```

---

### 3.3 Denoiser（去噪器）

**文件：`sgm/modules/diffusionmodules/denoiser.py`**

**作用：** 对 UNet 输出应用预条件，得到最终去噪样本

**核心代码 (行 23-39)：**
```python
def forward(self, network, input, sigma, cond, **kwargs):
    # 1. 计算预条件系数
    c_skip, c_out, c_in, c_noise = self.scaling(sigma)

    # 2. 调用 UNet
    network_output = network(input * c_in, c_noise, cond, **kwargs)

    # 3. 应用预条件
    return network_output * c_out + input * c_skip
```

**数学：** $\hat{z}_\theta = c_\text{skip} \cdot z_t + c_\text{out} \cdot F_\theta(c_\text{in} \cdot z_t, c_\text{noise})$

**Scaling 类型：**
- `EpsScaling` - 预测噪声（DDPM 风格）
- `VScaling` - 预测速度
- `EDMScaling` - Karras 等 (2022) 方案

**文件：** `sgm/modules/diffusionmodules/denoiser_scaling.py`

---

### 3.4 Model（UNet 网络）

**文件：`sgm/modules/diffusionmodules/openaimodel.py`**

**作用：** 原始神经网络，输出残差/预测

**包装器：** `sgm/modules/diffusionmodules/wrappers.py`
```python
class OpenAIWrapper(nn.Module):
    def forward(self, x, t, c, **kwargs):
        # 处理条件拼接
        x = torch.cat((x, c.get("concat", [])), dim=1)

        # 调用 UNet
        return self.diffusion_model(
            x,
            timesteps=t,              # c_noise (σ_t 或其变换)
            context=c.get("crossattn", None),  # 文本条件
            y=c.get("vector", None),           # 向量条件
            **kwargs
        )
```

**UNet 结构：**
- 时间嵌入 + 条件嵌入
- 编码器（下采样）
- 中间层
- 解码器（上采样，带跳跃连接）

---

### 3.5 Loss Function（损失函数）

**文件：`sgm/modules/diffusionmodules/loss.py`**

**核心训练逻辑 (行 59-90)：**
```python
def _forward(self, network, denoiser, cond, input, batch):
    # 1. 采样噪声级别
    sigmas = self.sigma_sampler(input.shape[0])  # σ_t ~ p(σ)

    # 2. 生成噪声
    noise = torch.randn_like(input)  # ε ~ N(0, I)

    # 3. 添加噪声
    sigmas_bc = append_dims(sigmas, input.ndim)
    noised_input = input + noise * sigmas_bc  # z_t = z_0 + σ_t * ε

    # 4. 模型预测
    model_output = denoiser(network, noised_input, sigmas, cond)

    # 5. 计算权重
    w = append_dims(self.loss_weighting(sigmas), input.ndim)

    # 6. 损失
    return w * (model_output - input) ** 2  # MSE
```

**数学：** $\mathcal{L} = \mathbb{E}[w(\sigma_t) \|\hat{z}_\theta(z_t, \sigma_t) - z_0\|^2]$

**组件：**
- `sigma_sampler` - 噪声级别采样器（EDMSampling, DiscreteSampling）
- `loss_weighting` - 损失权重（EDMWeighting, EpsWeighting, UnitWeighting）

---

### 3.6 Sigma Sampler（噪声级别采样）

**文件：`sgm/modules/diffusionmodules/sigma_sampling.py`**

**EDMSampling (行 6-13)：**
```python
class EDMSampling:
    def __init__(self, p_mean=-1.2, p_std=1.2):
        self.p_mean = p_mean
        self.p_std = p_std

    def __call__(self, n_samples):
        log_sigma = self.p_mean + self.p_std * torch.randn((n_samples,))
        return log_sigma.exp()
```

**数学：** $\log \sigma_t \sim \mathcal{N}(p_\text{mean}, p_\text{std}^2)$

---

### 3.7 Scheduler（学习率调度器）

**位置：** 配置文件中的 `scheduler_config`

**常用类型：**
- `LambdaLR` - 自定义学习率函数
- Cosine Annealing
- Step Decay

**使用：** `diffusion.py` 中通过 PyTorch Lightning 自动管理

---

## 4. 完整训练流程

```
main.py (启动)
    ↓
trainer.fit(model, data)
    ↓
DiffusionEngine.training_step(batch)
    ↓
DiffusionEngine.shared_step(batch)
    │
    ├─ get_input(batch) → x (图像)
    │
    ├─ encode_first_stage(x) → z_0 (VAE 编码)
    │
    └─ forward(z_0, batch)
         ↓
       loss_fn(model, denoiser, conditioner, z_0, batch)
         │
         ├─ conditioner(batch) → cond (条件编码)
         │
         ├─ sigma_sampler() → σ_t (采样噪声级别)
         │
         ├─ z_t = z_0 + σ_t * ε (添加噪声)
         │
         ├─ denoiser(model, z_t, σ_t, cond) → ẑ_θ
         │    │
         │    ├─ scaling(σ_t) → c_skip, c_out, c_in, c_noise
         │    │
         │    ├─ model(z_t * c_in, c_noise, cond) → F_θ
         │    │
         │    └─ return c_skip * z_t + c_out * F_θ
         │
         ├─ w = loss_weighting(σ_t) (计算权重)
         │
         └─ loss = w * ||ẑ_θ - z_0||² (MSE 损失)
```

---

## 5. 配置文件结构

**示例：** `configs/example_training/toy/mnist_diffusion.yaml`

```yaml
model:
  target: sgm.models.diffusion.DiffusionEngine
  params:
    # 网络
    network_config:
      target: sgm.modules.diffusionmodules.openaimodel.UNetModel

    # 去噪器
    denoiser_config:
      target: sgm.modules.diffusionmodules.denoiser.Denoiser
      params:
        scaling_config:
          target: sgm.modules.diffusionmodules.denoiser_scaling.EDMScaling

    # VAE
    first_stage_config:
      target: sgm.models.autoencoder.AutoencoderKL

    # 条件器
    conditioner_config:
      target: sgm.modules.GeneralConditioner

    # 损失函数
    loss_fn_config:
      target: sgm.modules.diffusionmodules.loss.StandardDiffusionLoss
      params:
        sigma_sampler_config:
          target: sgm.modules.diffusionmodules.sigma_sampling.EDMSampling
        loss_weighting_config:
          target: sgm.modules.diffusionmodules.denoiser_weighting.EDMWeighting

    # 调度器
    scheduler_config:
      target: torch.optim.lr_scheduler.LambdaLR

data:
  target: sgm.data.dataset.StableDataModuleFromConfig
  params:
    train:
      target: sgm.data.mnist.MNISTLoader
```

---

## 6. 核心组件总结表

| 组件 | 文件 | 作用 | 配置参数 |
|------|------|------|----------|
| **启动** | `main.py` | 训练入口 | - |
| **主模型** | `sgm/models/diffusion.py` | PyTorch Lightning 模块 | `model` |
| **VAE** | `sgm/models/autoencoder.py` | 图像 ↔ 潜在空间 | `first_stage_config` |
| **Conditioner** | `sgm/modules/encoders/modules.py` | 条件编码 | `conditioner_config` |
| **Denoiser** | `sgm/modules/diffusionmodules/denoiser.py` | 预条件 + 去噪 | `denoiser_config` |
| **Scaling** | `sgm/modules/diffusionmodules/denoiser_scaling.py` | 预条件系数 | `scaling_config` |
| **UNet** | `sgm/modules/diffusionmodules/openaimodel.py` | 神经网络 | `network_config` |
| **Wrapper** | `sgm/modules/diffusionmodules/wrappers.py` | 网络包装 | `network_wrapper` |
| **Loss** | `sgm/modules/diffusionmodules/loss.py` | 训练损失 | `loss_fn_config` |
| **Sigma Sampler** | `sgm/modules/diffusionmodules/sigma_sampling.py` | 噪声级别采样 | `sigma_sampler_config` |
| **Loss Weighting** | `sgm/modules/diffusionmodules/denoiser_weighting.py` | 损失权重 | `loss_weighting_config` |
| **Scheduler** | PyTorch 内置 | 学习率调度 | `scheduler_config` |

---

## 7. 如何替换为分数函数模型

### 7.1 核心转换

**数学公式：** $\hat{z}_\theta = z_t + \sigma_t^2 \cdot s_\theta$

其中 $s_\theta$ 是分数函数输出。

### 7.2 推荐方法：自定义 Scaling

**创建文件：** `sgm/modules/diffusionmodules/denoiser_scaling.py`（在现有文件中添加）

```python
class ScoreScaling:
    """用于分数函数模型的预条件"""
    def __call__(self, sigma):
        c_skip = torch.ones_like(sigma)   # 1
        c_out = sigma ** 2                 # σ²
        c_in = torch.ones_like(sigma)     # 1
        c_noise = sigma.clone()            # σ
        return c_skip, c_out, c_in, c_noise
```

**配置使用：**
```yaml
denoiser_config:
  target: sgm.modules.diffusionmodules.denoiser.Denoiser
  params:
    scaling_config:
      target: sgm.modules.diffusionmodules.denoiser_scaling.ScoreScaling
```

**效果：** 去噪器会自动将分数转换为去噪样本：
$$\hat{z}_\theta = 1 \cdot z_t + \sigma^2 \cdot s_\theta$$

### 7.3 替代方法：包装器

**创建文件：** `sgm/modules/diffusionmodules/score_wrapper.py`

```python
class ScoreToDenoised(nn.Module):
    def __init__(self, score_model):
        super().__init__()
        self.score_model = score_model

    def forward(self, x, timesteps, context=None, y=None, **kwargs):
        # 获取分数
        score = self.score_model(x, timesteps, context, y, **kwargs)

        # 转换为去噪样本
        sigma = append_dims(timesteps, x.ndim)
        denoised = x + (sigma ** 2) * score

        return denoised
```

**配置使用：**
```yaml
network_config:
  target: sgm.modules.diffusionmodules.wrappers.OpenAIWrapper
  params:
    diffusion_model:
      target: sgm.modules.diffusionmodules.score_wrapper.ScoreToDenoised
      params:
        score_model:
          target: your_module.YourScoreModel
```

---

## 8. 关键文件快速索引

| 需求 | 文件 | 行号 |
|------|------|------|
| 启动训练 | `main.py` | 900 |
| 训练步骤 | `sgm/models/diffusion.py` | 165-187 |
| VAE 编码 | `sgm/models/diffusion.py` | 138-150 |
| 前向扩散 | `sgm/modules/diffusionmodules/loss.py` | 84 |
| 模型预测 | `sgm/modules/diffusionmodules/loss.py` | 86-90 |
| 去噪器 | `sgm/modules/diffusionmodules/denoiser.py` | 23-39 |
| 预条件 | `sgm/modules/diffusionmodules/denoiser_scaling.py` | 全文 |
| UNet | `sgm/modules/diffusionmodules/openaimodel.py` | 866-903 |
| Sigma 采样 | `sgm/modules/diffusionmodules/sigma_sampling.py` | 6-13 |
