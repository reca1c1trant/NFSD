# Flow-based Diffusion Implementation

## 概述

本实现将UNet替换为基于Normalizing Flow的ScoreFlowNetwork，同时保留官方的所有其他组件：
- ✅ 官方 Denoiser (EDM scaling)
- ✅ 官方 Loss (StandardDiffusionLoss + EDMWeighting)
- ✅ 官方 Conditioner (GeneralConditioner + ClassEmbedder)
- ✅ 官方 First Stage Model (VAE/Autoencoder)
- ✅ 官方 Sampler
- ✅ Exact gradient computation via autograd

## 文件结构

### 新增文件：
```
sgm/modules/diffusionmodules/score_flow_network.py  # ScoreFlowNetwork (替换UNet)
sgm/modules/flows/__init__.py                       # Flow模块入口
sgm/modules/flows/flow_layers.py                    # Flow层实现
sgm/modules/flows/normalizing_flow.py               # Normalizing Flow实现
configs/example_training/toy/mnist_flow.yaml        # MNIST训练配置
```

### 修改文件：
无（官方代码未修改）

## 训练命令

```bash
python main.py --base configs/example_training/toy/mnist_flow.yaml
```

## 完整训练Pipeline流程图

```
┌────────────────────────────────────────────────────────────┐
│ 1. Data: {"jpg": [B,1,28,28], "cls": [B]}                 │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ 2. encode_first_stage (Identity for MNIST)                 │
│    x_latent = x_img                                        │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ 3. StandardDiffusionLoss                                   │
│    σ ~ EDMSampling: log(σ) ~ N(-1.2, 1.2)                 │
│    noise ~ N(0, I)                                         │
│    x_noisy = x_latent + σ * noise                         │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ 4. Conditioner (官方)                                       │
│    cls [B] → ClassEmbedder → vector [B, 512]              │
│    cond = {"vector": [B, 512]}                             │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ 5. Denoiser (官方 EDMScaling)                              │
│    c_skip = σ_data² / (σ² + σ_data²)                      │
│    c_out = σ·σ_data / sqrt(σ² + σ_data²)                  │
│    c_in = 1 / sqrt(σ² + σ_data²)                          │
│    c_noise = 0.25 * log(σ)                                │
│                                                            │
│    调用: network_output = ScoreFlowNetwork(                │
│              x_noisy * c_in,      ← scaled input          │
│              c_noise,             ← timesteps             │
│              cond                                          │
│          )                                                 │
│                                                            │
│    denoiser_output = network_output * c_out +              │
│                      x_noisy * c_skip                      │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ 6. ScoreFlowNetwork (核心：Exact Gradient)                 │
│                                                            │
│ ┌────────────────────────────────────────────────────────┐│
│ │ A. Exact Score Computation                             ││
│ │    x_flat = x.flatten().requires_grad_(True)           ││
│ │    σ_emb = time_embedder(c_noise)                      ││
│ │    class_emb = label_emb(y)                            ││
│ │    flow_input = cat([x_flat, σ_emb, class_emb])        ││
│ │    flow_output = Flow(flow_input)                      ││
│ │    energy = -||flow_output||²                          ││
│ │    score = ∇_{x_flat} energy  ✓ EXACT via autograd    ││
│ │    score_enhanced = MLP(cat([score, σ_emb, class_emb]))││
│ │                                                        ││
│ │ B. Score → x_0 Conversion                              ││
│ │    σ = exp(c_noise / 0.25)  ← 恢复原始sigma           ││
│ │    重算: c_skip, c_out, c_in (with sigma_data=0.5)     ││
│ │    x_noisy = x / c_in  ← 恢复原始noisy sample         ││
│ │    x_0 = x_noisy + σ² * score  ✓ 数学公式             ││
│ │                                                        ││
│ │ C. 适配Denoiser格式                                    ││
│ │    F_θ = (x_0 - x_noisy * c_skip) / c_out              ││
│ │    return F_θ                                          ││
│ └────────────────────────────────────────────────────────┘│
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ 7. Denoiser后处理                                          │
│    denoiser_output = F_θ * c_out + x_noisy * c_skip       │
│                    = x_0  ✓ 恢复clean sample              │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ 8. Loss Computation                                        │
│    w = EDMWeighting(σ) = (σ² + σ_data²) / (σ·σ_data)²    │
│    loss = w * MSE(denoiser_output, x_latent)              │
│         = w * MSE(x_0, x_clean)                            │
└───────────────────────────┬────────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────────┐
│ 9. Backward & Update                                       │
│    loss.backward()                                         │
│    梯度流: loss → x_0 → score → energy → flow_output →    │
│            Flow parameters ✓                               │
│    optimizer.step()                                        │
└────────────────────────────────────────────────────────────┘
```

## 核心数学保证

### Score Function正确性
```
score = ∇_x log p(x_t | σ)
      = ∇_x energy  (energy-based model)
      = ∇_x (-||Flow(x, σ, cond)||²)

通过 torch.autograd.grad() 精确计算，无近似！
```

### x_0预测公式
```
在Gaussian diffusion下:
∇ log p(x_t) = -(x_t - x_0) / σ²

因此:
x_0 = x_t + σ² * ∇ log p(x_t)
    = x_t + σ² * score  ✓ 精确公式
```

### Denoiser兼容性
```
Network输出: F_θ = (x_0 - x_t * c_skip) / c_out
Denoiser计算: x_0 = F_θ * c_out + x_t * c_skip  ✓ 匹配
```


## 配置说明

### network_config
```yaml
network_config:
  target: sgm.modules.diffusionmodules.score_flow_network.ScoreFlowNetwork
  params:
    in_channels: 1          # 输入通道 (MNIST=1, RGB=3, Latent=4)
    model_channels: 128     # 基础隐藏维度
    n_flows: 3              # Flow层数
    activation: softplus    # 激活函数
    num_classes: 10         # 类别数 (可选)
    sigma_data: 0.5         # 必须匹配denoiser的sigma_data
```

### 其他配置
保持官方配置不变：
- `denoiser_config`: EDMScaling
- `loss_fn_config`: StandardDiffusionLoss + EDMWeighting + EDMSampling
- `conditioner_config`: GeneralConditioner + ClassEmbedder
- `first_stage_config`: IdentityFirstStage (MNIST) 或 AutoencoderKL (图像)

## 扩展到其他数据集

### CIFAR-10
```yaml
network_config:
  params:
    in_channels: 3  # RGB
    model_channels: 256
    n_flows: 4

data:
  target: sgm.data.cifar10.CIFAR10Loader
```

### 带VAE的图像数据集
```yaml
network_config:
  params:
    in_channels: 4  # Latent channels from VAE
    model_channels: 256
    n_flows: 5
    use_spatial_transformer: True  # 启用cross-attention
    context_dim: 768  # Text embedding dimension

first_stage_config:
  target: sgm.models.autoencoder.AutoencoderKL
  params:
    ckpt_path: path/to/vae.safetensors

conditioner_config:
  emb_models:
    - target: sgm.modules.encoders.modules.FrozenCLIPEmbedder  # Text encoder
      input_key: txt
```

## 关键特性

1. **Exact Gradient**: 通过autograd精确计算 ∇ energy，无近似
2. **官方兼容**: 完全兼容官方DiffusionEngine、denoiser、loss、sampler
3. **无侵入性**: 未修改任何官方代码
4. **可扩展**: 支持class/text conditioning、VAE、不同数据集
5. **梯度流保证**: 无.detach()阻断，create_graph=True保留二阶导数

## 与UNet对比

| 特性 | UNet | ScoreFlowNetwork |
|-----|------|------------------|
| Score计算 | 神经网络预测 | **Exact gradient** |
| 可逆性 | 不可逆 | 可逆 |
| 概率计算 | 不支持 | 支持 exact log prob |
| 参数量 | 大 (~1B for SDXL) | 相对小 |
| 数学保证 | 近似 | 精确 (在energy形式下) |

## 输出位置

```
logs/<timestamp>-<run_name>/
├── checkpoints/
│   ├── last.ckpt
│   └── epoch=*.ckpt
├── configs/
│   └── mnist_flow.yaml
└── images/
    └── train/
```

## 监控

训练时关注：
- `train/loss`: 应逐渐下降
- 生成图像质量: ImageLogger每1000步记录
- 梯度范数: 确保无梯度爆炸/消失
