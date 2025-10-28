# Flow-based Diffusion 快速说明

## ✅ 完成状态

**9 个新文件已创建，0 个现有文件被修改**

---

## 📂 新增文件

### 核心代码（4 个）
- `sgm/modules/flows/__init__.py`
- `sgm/modules/flows/flow_layers.py`
- `sgm/modules/flows/normalizing_flow.py`
- `sgm/models/score_flow.py`

### 配置（2 个）
- `configs/training/flow_diffusion_mnist.yaml`
- `configs/training/flow_diffusion_toy.yaml`

### 脚本（2 个）
- `scripts/train_flow_diffusion.py`
- `scripts/sampling/simple_flow_sample.py`

### 文档（3 个）
- `FLOW_DIFFUSION_GUIDE.md` - 完整使用指南 ⭐
- `architecture_redesign.md` - 架构设计文档
- `IMPLEMENTATION_SUMMARY.md` - 实现总结

---

## 🚀 快速开始

### 1. 训练模型

```bash
python scripts/train_flow_diffusion.py \
    --config configs/training/flow_diffusion_mnist.yaml \
    --name my_first_flow
```

### 2. 生成样本

```bash
python scripts/sampling/simple_flow_sample.py \
    --config configs/training/flow_diffusion_mnist.yaml \
    --checkpoint logs/[timestamp]-my_first_flow/checkpoints/last.ckpt \
    --num_samples 16
```

---

## 🎯 核心原理

**用 Normalizing Flow 替代 UNet**

```
原架构: UNet → x̂ → score = (x̂ - x_t)/σ²
新架构: Flow → ∇log p(x_t) = score → x̂ = x_t + σ²·score
```

- ✅ 参数量：数百万（vs UNet 的数十亿）
- ✅ 理论：精确概率建模
- ✅ Score：直接通过梯度计算

---

## 📖 详细文档

请查看 **`FLOW_DIFFUSION_GUIDE.md`** 获取：
- 详细使用教程
- 配置参数说明
- 常见问题解答
- 高级用法示例

---

## ⚡ 关键特性

1. **完全独立** - 不修改任何现有文件
2. **即开即用** - 配置和脚本开箱即用
3. **易扩展** - 支持多种激活函数和配置
4. **兼容性** - 使用相同的训练框架

---

**开始实验吧！** 🎉
