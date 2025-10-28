# 去噪样本与分数函数的数学联系

## 核心结论

**是的，去噪样本和分数函数在数学上是等价的表示。** 你可以通过数学变换相互转换，现代扩散模型通过预测去噪样本来隐式计算分数函数。本报告基于 Stability AI 代码库，详细解释二者的数学联系。

---

## 1. 背景：基于分数的扩散模型

### 1.1 分数函数定义

在基于分数的生成模型中，**分数函数**定义为：

$$\mathbf{s}_\theta(\mathbf{x}_t, t) = \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$$

这表示对数概率密度相对于噪声数据 $\mathbf{x}_t$ 的梯度。

### 1.2 前向扩散过程

前向过程添加高斯噪声：

$$\mathbf{x}_t = \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

其中：
- $\mathbf{x}_0$ 是干净数据
- $\mathbf{x}_t$ 是噪声级别 $\sigma_t$ 下的噪声数据
- $\boldsymbol{\epsilon}$ 是标准高斯噪声

---

## 2. 三种等价的参数化方式

现代扩散模型可以预测以下三个量中的任意一个，它们在数学上是等价的：

### 2.1 噪声预测（ε-prediction）
预测噪声：$\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \sigma_t)$

### 2.2 数据预测（x₀-prediction / 去噪预测）
预测干净数据：$\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t)$

### 2.3 分数预测
预测分数：$\mathbf{s}_\theta(\mathbf{x}_t, \sigma_t)$

---

## 3. 核心数学关系

### 3.1 从去噪样本到分数函数

**关键公式：**
$$\boxed{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \frac{\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t) - \mathbf{x}_t}{\sigma_t^2}}$$

**推导过程：**

给定前向过程 $\mathbf{x}_t = \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$，条件分布为：

$$p_t(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \mathbf{x}_0, \sigma_t^2 \mathbf{I})$$

该高斯分布的分数为：
$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t | \mathbf{x}_0) = \frac{\mathbf{x}_0 - \mathbf{x}_t}{\sigma_t^2}$$

根据 Tweedie 公式（贝叶斯后验均值），最优去噪器为：
$$\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t) = \mathbb{E}[\mathbf{x}_0 | \mathbf{x}_t] = \mathbf{x}_t + \sigma_t^2 \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$$

重新整理得到：
$$\boxed{\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = \frac{\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t) - \mathbf{x}_t}{\sigma_t^2}}$$

### 3.2 从分数函数到去噪样本

重新整理上述方程：
$$\boxed{\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t) = \mathbf{x}_t + \sigma_t^2 \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)}$$

这就是 **Tweedie 公式** —— 干净数据的后验均值估计。

### 3.3 噪声预测的联系

从 $\mathbf{x}_t = \mathbf{x}_0 + \sigma_t \boldsymbol{\epsilon}$ 得到：

$$\boldsymbol{\epsilon} = \frac{\mathbf{x}_t - \mathbf{x}_0}{\sigma_t} = \frac{\mathbf{x}_t - \hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t)}{\sigma_t}$$

因此：
$$\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t) = \mathbf{x}_t - \sigma_t \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \sigma_t)$$

分数为：
$$\nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \sigma_t)}{\sigma_t}$$

### 3.4 转换公式总结

| 从 | 到 | 公式 |
|---|---|------|
| 去噪样本 | 分数 | $\mathbf{s}_\theta = \frac{\hat{\mathbf{x}}_\theta - \mathbf{x}_t}{\sigma_t^2}$ |
| 分数 | 去噪样本 | $\hat{\mathbf{x}}_\theta = \mathbf{x}_t + \sigma_t^2 \mathbf{s}_\theta$ |
| 去噪样本 | 噪声 | $\boldsymbol{\epsilon}_\theta = \frac{\mathbf{x}_t - \hat{\mathbf{x}}_\theta}{\sigma_t}$ |
| 噪声 | 去噪样本 | $\hat{\mathbf{x}}_\theta = \mathbf{x}_t - \sigma_t \boldsymbol{\epsilon}_\theta$ |
| 噪声 | 分数 | $\mathbf{s}_\theta = -\frac{\boldsymbol{\epsilon}_\theta}{\sigma_t}$ |
| 分数 | 噪声 | $\boldsymbol{\epsilon}_\theta = -\sigma_t \mathbf{s}_\theta$ |

---

## 4. 模型实际输出是什么？

### 4.1 关键发现：两阶段架构

**答案：原始神经网络（UNet）并不直接输出去噪样本！**

完整的管道���**两个阶段**：

1. **原始网络输出（UNet）**：输出残差或预测值 $F_\theta(\mathbf{x}_t, \sigma_t)$（取决于缩放方案）
2. **去噪器包装器（Denoiser）**：应用预条件生成**最终去噪样本** $\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t)$

### 4.2 去噪器公式

**代码位置：** `sgm/modules/diffusionmodules/denoiser.py:36-39`

```python
return (
    network(input * c_in, c_noise, cond, **additional_model_inputs) * c_out
    + input * c_skip
)
```

**数学公式：**
$$\boxed{\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t) = c_\text{skip}(\sigma_t) \cdot \mathbf{x}_t + c_\text{out}(\sigma_t) \cdot F_\theta(c_\text{in}(\sigma_t) \cdot \mathbf{x}_t, c_\text{noise}(\sigma_t))}$$

其中：
- $\mathbf{x}_t$ = 噪声输入
- $F_\theta$ = 原始网络输出（UNet）
- $c_\text{skip}, c_\text{out}, c_\text{in}, c_\text{noise}$ = 预条件系数（依赖于 $\sigma_t$）
- $\hat{\mathbf{x}}_\theta$ = **最终去噪输出**

### 4.3 三种缩放方案

#### A. EpsScaling（DDPM 风格）

**使用模型：** SDXL、SD 2.1、训练配置

**代码：** `sgm/modules/diffusionmodules/denoiser_scaling.py:29-37`
```python
c_skip = torch.ones_like(sigma)
c_out = -sigma
c_in = 1 / (sigma**2 + 1.0) ** 0.5
c_noise = sigma.clone()
```

**公式：**
$$\hat{\mathbf{x}}_\theta = \mathbf{x}_t - \sigma_t \cdot F_\theta\left(\frac{\mathbf{x}_t}{\sqrt{\sigma_t^2 + 1}}, \sigma_t\right)$$

**解释：**
- $F_\theta$ 预测**噪声** $\boldsymbol{\epsilon}$
- 去噪样本 = 噪声输入 - 噪声预测 × 噪声级别

#### B. VScaling（速度预测）

**使用模型：** SV3D、SVD、SD 2.1-768

**代码：** `sgm/modules/diffusionmodules/denoiser_scaling.py:40-48`
```python
c_skip = 1.0 / (sigma**2 + 1.0)
c_out = -sigma / (sigma**2 + 1.0) ** 0.5
c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
c_noise = sigma.clone()
```

**公式：**
$$\hat{\mathbf{x}}_\theta = \frac{1}{\sigma_t^2 + 1} \mathbf{x}_t - \frac{\sigma_t}{\sqrt{\sigma_t^2 + 1}} F_\theta\left(\frac{\mathbf{x}_t}{\sqrt{\sigma_t^2 + 1}}, \sigma_t\right)$$

**解释：**
- $F_\theta$ 预测**速度**（数据和噪声的组合）
- 在极端噪声级别下数值更稳定

#### C. EDMScaling（Karras 等，2022）

**使用模型：** 玩具示例（MNIST、CIFAR-10）

**代码：** `sgm/modules/diffusionmodules/denoiser_scaling.py:15-26`
```python
c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
c_noise = 0.25 * sigma.log()
```

**公式：**
$$\hat{\mathbf{x}}_\theta = \frac{\sigma_\text{data}^2}{\sigma_t^2 + \sigma_\text{data}^2} \mathbf{x}_t + \frac{\sigma_t \sigma_\text{data}}{\sqrt{\sigma_t^2 + \sigma_\text{data}^2}} F_\theta\left(\frac{\mathbf{x}_t}{\sqrt{\sigma_t^2 + \sigma_\text{data}^2}}, \frac{\ln \sigma_t}{4}\right)$$

**解释：**
- $F_\theta$ 预测**缩放残差**
- 在所有噪声级别下有更好的条件数

### 4.4 完整数据流

```
训练流程：
┌─────────────┐
│   x_0       │ 干净数据
│  (目标)      │
└──────┬──────┘
       │ + 噪声
       ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   x_t       │────▶│  原始网络     │────▶│   F_θ       │
│  (噪声)      │     │   (UNet)     │     │  (残差)      │
└─────────────┘     └──────────────┘     └──────┬──────┘
       │                                         │
       │            ┌──────────────┐            │
       └───────────▶│   去噪器      │◀───────────┘
                    │ (预条件系数)  │
                    └──────┬───────┘
                           ▼
                    ┌─────────────┐
                    │  x̂_θ        │ 去噪预测
                    │  (输出)      │
                    └──────┬──────┘
                           │
                           ▼
                    损失: ||x̂_θ - x_0||²

推理流程（相同管道）：
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  x_t        │────▶│  原始网络     │────▶│   F_θ       │
│  (采样)      │     │   (UNet)     │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
       │                                         │
       │            ┌──────────────┐            │
       └───────────▶│   去噪器      │◀───────────┘
                    └──────┬───────┘
                           ▼
                    ┌─────────────┐
                    │  x̂_θ        │ 去噪样本（采样器使用）
                    └─────────────┘
```

### 4.5 模型输出总结

| 组件 | 输出 | 公式 | 含义 |
|-----|------|------|------|
| **原始网络（UNet）** | $F_\theta(\mathbf{x}_t, \sigma_t)$ | `self.out(h)` | 残差/预测（取决于缩放） |
| **去噪器包装器** | $\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t)$ | $c_\text{skip} \mathbf{x}_t + c_\text{out} F_\theta$ | **最终去噪样本** |

**直接回答：**

❌ **否** - 原始神经网络不直接输出去噪样本

✅ **是** - 去噪器包装器（网络 + 预条件）输出去噪样本

**代码中看到的"model_output"指的是完整管道输出（去噪器），它就是去噪样本 $\hat{\mathbf{x}}_\theta$。**

---

## 5. Stability AI 代码库证据

### 5.1 训练：模型预测去噪样本

**文件：** `sgm/modules/diffusionmodules/loss.py:86-90`

```python
model_output = denoiser(
    network, noised_input, sigmas, cond, **additional_model_inputs
)
w = append_dims(self.loss_weighting(sigmas), input.ndim)
return self.get_loss(model_output, input, w)
```

**分析：**
- `noised_input` = $\mathbf{x}_t$（噪声数据）
- `input` = $\mathbf{x}_0$（干净数据，目标）
- `model_output` = $\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t)$（去噪预测）
- 损失：$\mathcal{L} = w(\sigma_t) \|\hat{\mathbf{x}}_\theta - \mathbf{x}_0\|^2$

**结论：模型被训练来预测去噪样本（干净数据），而不是直接预测分数。**

### 5.2 推理：将去噪样本转换为导数

**文件：** `sgm/modules/diffusionmodules/sampling_utils.py:34-35`

```python
def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)
```

**数学解释：**

$$d = \frac{\mathbf{x}_t - \hat{\mathbf{x}}_\theta}{\sigma_t}$$

这在 ODE/SDE 求解器中使用。对于概率流 ODE：

$$\frac{d\mathbf{x}}{dt} = -\frac{1}{2}\sigma_t \frac{d\sigma_t}{dt} \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t)$$

代入我们的转换：

$$d = \frac{\mathbf{x}_t - \hat{\mathbf{x}}_\theta}{\sigma_t} = \frac{\mathbf{x}_t - (\mathbf{x}_t + \sigma_t^2 \nabla \log p_t)}{\sigma_t} = -\sigma_t \nabla \log p_t$$

因此：$d = -\sigma_t \mathbf{s}_\theta(\mathbf{x}_t, \sigma_t)$

**这证实了 `to_d` 隐式计算了与分数成正比的量！**

### 5.3 采样循环使用去噪预测

**文件：** `sgm/modules/diffusionmodules/sampling.py:93-107`（EDM 采样器）

```python
def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
    sigma_hat = sigma * (gamma + 1.0)
    if gamma > 0:
        eps = torch.randn_like(x) * self.s_noise
        x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

    denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
    d = to_d(x, sigma_hat, denoised)  # 将去噪转换为导数
    dt = append_dims(next_sigma - sigma_hat, x.ndim)

    euler_step = self.euler_step(x, d, dt)
    # ... 可能的修正步骤
    return x
```

**流程：**
1. 获取 `denoised` = $\hat{\mathbf{x}}_\theta(\mathbf{x}_t, \sigma_t)$
2. 计算 `d = to_d(x, sigma, denoised)` = 与分数成正比的导数
3. 欧拉步骤：$\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + d \cdot \Delta t$

**采样器使用去噪预测，并通过 `to_d` 隐式计算分数。**

---

## 6. 关键问题：如果新模型直接输出分数函数，能否通过数学变换获得相同结果？

### 6.1 答案：是的，完全可以！

**理论保证：** 由于去噪样本和分数函数在数学上是等价的，直接预测分数函数的模型可以通过简单的数学变换获得与预测去噪样本的模型完全相同的结果。

### 6.2 两种方法的对比

#### 方法 A：当前 Stability AI 方法（预测去噪样本）

**训练：**
```python
# 训练目标
model_output = denoiser(network, x_t, sigma, cond)  # 输出去噪样本
loss = ||model_output - x_0||²
```

**推理：**
```python
# 采样步骤
denoised = denoiser(network, x_t, sigma, cond)      # 获取去噪样本
d = (x_t - denoised) / sigma                        # 转换为导数
x_next = x_t + d * dt                                # ODE 步骤
```

#### 方法 B：直接预测分数函数

**训练：**
```python
# 训练目标
score_output = network(x_t, sigma, cond)            # 直接输出分数
true_score = (x_0 - x_t) / sigma**2                 # 真实分数
loss = ||score_output - true_score||²
```

**推理选项 1：直接使用分数**
```python
# 采样步骤
score = network(x_t, sigma, cond)                   # 获取分数
d = -sigma * score                                  # 转换为导数
x_next = x_t + d * dt                                # ODE 步骤
```

**推理选项 2：转换为去噪样本**
```python
# 采样步骤
score = network(x_t, sigma, cond)                   # 获取分数
denoised = x_t + sigma**2 * score                   # 转换为去噪样本
d = (x_t - denoised) / sigma                        # 转换为导数
x_next = x_t + d * dt                                # ODE 步骤
```

### 6.3 数学等价性证明

**方法 A 的导数：**
$$d_A = \frac{\mathbf{x}_t - \hat{\mathbf{x}}_\theta}{\sigma_t}$$

**方法 B 选项 1 的导数：**
$$d_B = -\sigma_t \mathbf{s}_\theta$$

**由于 Tweedie 公式：**
$$\hat{\mathbf{x}}_\theta = \mathbf{x}_t + \sigma_t^2 \mathbf{s}_\theta$$

**代入方法 A：**
$$d_A = \frac{\mathbf{x}_t - (\mathbf{x}_t + \sigma_t^2 \mathbf{s}_\theta)}{\sigma_t} = \frac{-\sigma_t^2 \mathbf{s}_\theta}{\sigma_t} = -\sigma_t \mathbf{s}_\theta = d_B$$

**结论：$d_A = d_B$ ，两种方法产生完全相同的采样轨迹！**

### 6.4 实际实现：如何修改现有代码

如果你想构建一个直接输出分数函数的模型，可以这样修改：

#### 修改 1：自定义缩放（ScoreScaling）

创建新文件或修改 `sgm/modules/diffusionmodules/denoiser_scaling.py`：

```python
class ScoreScaling:
    """直接预测分数函数的缩放方案"""
    def __call__(self, sigma: torch.Tensor):
        # 网络输出分数，我们需要转换为去噪样本
        # denoised = x_t + sigma^2 * score
        # 即：denoised = c_skip * x_t + c_out * network_output
        # 其中 network_output = score
        c_skip = torch.ones_like(sigma)           # x_t 的系数
        c_out = sigma**2                           # score 的系数
        c_in = torch.ones_like(sigma)             # 输入不需要缩放
        c_noise = sigma.clone()                    # 噪声级别
        return c_skip, c_out, c_in, c_noise
```

**公式：**
$$\hat{\mathbf{x}}_\theta = 1 \cdot \mathbf{x}_t + \sigma_t^2 \cdot F_\theta(\mathbf{x}_t, \sigma_t)$$

其中 $F_\theta$ 输出分数 $\mathbf{s}_\theta$。

#### 修改 2：训练损失

修改 `sgm/modules/diffusionmodules/loss.py`：

```python
class ScoreDiffusionLoss(StandardDiffusionLoss):
    """基于分数匹配的损失"""

    def get_loss(self, model_output, target, w):
        # model_output 是去噪样本（从分数转换而来）
        # target 是 x_0
        # 但我们希望训练分数，所以需要转换

        # 从去噪样本反推分数
        # denoised = x_t + sigma^2 * score
        # 我们需要计算 score 的损失

        # 实际上，由于 denoiser 的设计，model_output 已经是去噪样本
        # 训练 ||denoised - x_0||^2 等价于训练分数匹配

        if self.loss_type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        # ... 其他损失类型
```

**或者，直接训练分数匹配损失：**

```python
class DirectScoreMatchingLoss(nn.Module):
    """直接分数匹配损失"""

    def forward(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        sigmas = self.sigma_sampler(input.shape[0]).to(input)
        noise = torch.randn_like(input)
        sigmas_bc = append_dims(sigmas, input.ndim)
        noised_input = input + noise * sigmas_bc

        # 获取模型输出（去噪样本）
        model_output = denoiser(network, noised_input, sigmas, cond)

        # 从去噪样本计算隐式分数
        pred_score = (model_output - noised_input) / (sigmas_bc ** 2)

        # 真实分数
        true_score = (input - noised_input) / (sigmas_bc ** 2)

        # 分数匹配损失
        loss = torch.mean((pred_score - true_score) ** 2, dim=list(range(1, len(pred_score.shape))))
        return loss
```

#### 修改 3：配置文件

在 `configs/` 中创建新配置：

```yaml
denoiser_config:
  target: sgm.modules.diffusionmodules.denoiser.Denoiser
  params:
    scaling_config:
      target: sgm.modules.diffusionmodules.denoiser_scaling.ScoreScaling
```

### 6.5 为什么现有方法不直接预测分数？

尽管数学上等价，预测去噪样本有实际优势：

#### 优势 1：数值稳定性

**分数预测：**
- 当 $\sigma_t \to 0$（低噪声）：$\mathbf{s}_\theta = \frac{\hat{\mathbf{x}}_\theta - \mathbf{x}_t}{\sigma_t^2} \to$ 非常大的值
- 当 $\sigma_t \to \infty$（高噪声）：$\mathbf{s}_\theta = \frac{\hat{\mathbf{x}}_\theta - \mathbf{x}_t}{\sigma_t^2} \to$ 非常小的值
- 网络需要输出跨越多个数量级的值

**去噪样本预测：**
- $\hat{\mathbf{x}}_\theta$ 始终在数据范围内（例如，图像为 [-1, 1]）
- 网络输出范围稳定
- 预条件（$c_\text{skip}, c_\text{out}$）自动处理不同噪声级别的缩放

#### 优势 2：训练简单

**分数预测：**
- 需要仔细设计权重 $w(\sigma_t)$ 来平衡不同噪声级别
- 分数的真实值 $\frac{\mathbf{x}_0 - \mathbf{x}_t}{\sigma_t^2}$ 在不同 $\sigma_t$ 下变化很大

**去噪样本预测：**
- 简单的 MSE 损失：$\|\hat{\mathbf{x}}_\theta - \mathbf{x}_0\|^2$
- 目标 $\mathbf{x}_0$ 在所有噪声级别下都相同
- 更容易训练和调试

#### 优势 3：可解释性

**分数预测：**
- 输出是梯度场（难以直接可视化）

**去噪样本预测：**
- 输出是干净图像的估计（容易可视化和理解）
- 可以在训练期间直接查看去噪质量

### 6.6 何时使用分数预测？

在某些情况下，直接预测分数可能更好：

1. **理论研究**：研究分数匹配和分数基础扩散模型的理论性质
2. **特殊架构**：某些神经网络架构天然适合输出梯度/分数
3. **连续时间模型**：某些连续时间公式更自然地使用分数
4. **与其他分数模型集成**：如果需要与其他基于分数的系统互操作

### 6.7 完整转换示例

假设你有一个训练好的模型，输出去噪样本 $\hat{\mathbf{x}}_\theta$，想要获取分数：

```python
def get_score_from_denoised_model(denoiser, network, x_t, sigma, cond):
    """
    从去噪模型获取分数函数

    Args:
        denoiser: 去噪器包装器
        network: UNet 网络
        x_t: 噪声样本
        sigma: 噪声级别
        cond: 条件

    Returns:
        score: 分数函数 ∇log p(x_t)
    """
    # 获取去噪样本
    denoised = denoiser(network, x_t, sigma, cond)

    # 转换为分数
    score = (denoised - x_t) / (sigma ** 2)

    return score


def get_denoised_from_score_model(score_network, x_t, sigma, cond):
    """
    从分数模型获取去噪样本

    Args:
        score_network: 直接输出分数的网络
        x_t: 噪声样本
        sigma: 噪声级别
        cond: 条件

    Returns:
        denoised: 去噪样本
    """
    # 获取分数
    score = score_network(x_t, sigma, cond)

    # 转换为去噪样本
    denoised = x_t + (sigma ** 2) * score

    return denoised
```

### 6.8 与现有采样器的兼容性

**好消息：** 所有现有的采样器（Euler、Heun、DPM++等）都可以无缝使用任一表示！

```python
# 现有采样器代码（sampling.py）
def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None):
    denoised = self.denoise(x, denoiser, sigma, cond, uc)
    d = to_d(x, sigma, denoised)
    dt = append_dims(next_sigma - sigma, x.ndim)
    return self.euler_step(x, d, dt)
```

只要 `denoiser` 返回去噪样本（无论内部如何计算），采样器都能正常工作：

- **方法 A**：`denoiser` 内部预测去噪样本 → 直接返回
- **方法 B**：`denoiser` 内部预测分数 → 转换为去噪样本 → 返回

采样器看到的都是去噪样本，因此产生相同的结果。

---

## 7. 总结表

| 方面 | 预测去噪样本 | 预测分数函数 |
|-----|------------|-------------|
| **数学等价性** | ✅ 完全等价 | ✅ 完全等价 |
| **采样结果** | ✅ 相同 | ✅ 相同（通过转换） |
| **数值稳定性** | ✅ 优秀（输出在数据范围内） | ⚠️ 一般（输出跨越多个数量级） |
| **训练简单性** | ✅ 简单（MSE 损失） | ⚠️ 需要仔细的权重设计 |
| **可解释性** | ✅ 易于可视化 | ⚠️ 梯度场难以解释 |
| **当前使用** | ✅ Stability AI、大多数现代模型 | 理论研究，早期论文 |
| **灵活性** | ✅ 多种预条件方案（EDM、Eps、V） | ✅ 可以使用相同的预条件 |
| **转换成本** | 低（仅需简单除法） | 低（仅需简单乘法） |

---

## 8. 最终结论

### 8.1 关于模型输出

**模型实际输出的是去噪样本 $\hat{\mathbf{x}}_\theta$，它是通过以下两阶段过程产生的：**

1. **UNet 网络** → 输出残差/预测 $F_\theta$
2. **去噪器包装器** → 应用预条件得到 $\hat{\mathbf{x}}_\theta = c_\text{skip} \mathbf{x}_t + c_\text{out} F_\theta$

### 8.2 关于等价性

**去噪样本和分数函数通过 Tweedie 公式连接：**

$$\hat{\mathbf{x}}_\theta = \mathbf{x}_t + \sigma_t^2 \mathbf{s}_\theta$$

**这意味着：**
- ✅ 可以从去噪样本计算分数：$\mathbf{s}_\theta = \frac{\hat{\mathbf{x}}_\theta - \mathbf{x}_t}{\sigma_t^2}$
- ✅ 可以从分数计算去噪样本：$\hat{\mathbf{x}}_\theta = \mathbf{x}_t + \sigma_t^2 \mathbf{s}_\theta$
- ✅ 两种表示在数学上完全等价
- ✅ 使用任一表示都能获得相同的采样结果

### 8.3 关于新模型输出分数函数

**如果新模型直接输出分数函数，回答是肯定的：**

✅ **是的，可以通过简单的数学变换获得与预测去噪样本的模型完全相同的结果！**

**所需变换：**
```python
# 如果模型输出分数
score = model(x_t, sigma, cond)

# 转换为去噪样本（用于采样器）
denoised = x_t + sigma**2 * score

# 之后的采样步骤完全相同
d = (x_t - denoised) / sigma
x_next = x_t + d * dt
```

**关键点：**
1. 两种方法产生数学上相同的采样轨迹
2. 可以创建自定义 `ScoreScaling` 类来适配现有框架
3. 所有现��采样器（Euler、Heun、DPM++）都兼容
4. 唯一的区别是训练稳定性和可解释性

### 8.4 实践建议

**如果你要构建新模型：**
- 对于实际应用：推荐预测去噪样本（更稳定、更易训练）
- 对于理论研究：预测分数函数也可以（更接近理论公式）
- 如果不确定：使用预条件框架（可以灵活切换）

**如果你有现有的分数模型：**
- 使用 `denoised = x_t + sigma**2 * score` 转换
- 可以无缝集成到 Stability AI 的采样框架中
- 采样结果将与去噪模型相同

---

## 参考文献

1. **Tweedie 公式：** Efron, B. (2011). "Tweedie's formula and selection bias." *JASA*.
2. **分数匹配：** Hyvärinen, A. (2005). "Estimation of non-normalized statistical models by score matching."
3. **去噪扩散：** Ho et al. (2020). "Denoising Diffusion Probabilistic Models."
4. **EDM 框架：** Karras et al. (2022). "Elucidating the Design Space of Diffusion-Based Generative Models."
5. **代码参考：** Stability AI generative-models 仓库
   - `sgm/modules/diffusionmodules/denoiser.py`
   - `sgm/modules/diffusionmodules/denoiser_scaling.py`
   - `sgm/modules/diffusionmodules/sampling_utils.py`
   - `sgm/modules/diffusionmodules/loss.py`
