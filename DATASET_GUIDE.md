# 数据集导入完整指南

## 📊 完整调用链

```
配置文件 (YAML)
    ↓
instantiate_from_config()
    ↓
DataModule (__init__)
    ↓
train_dataloader()
    ↓
PyTorch DataLoader
    ↓
trainer.fit(model, data)
    ↓
每个 batch: {"jpg": images, "cls": labels}
```

---

## 🔍 详细代码调用流程

### 1. 配置文件定义

**文件：** `configs/training/flow_diffusion_mnist.yaml`

```yaml
data:
  target: sgm.data.mnist.MNISTLoader  # 完整类路径
  params:
    batch_size: 256
    num_workers: 4
```

### 2. 训练脚本实例化

**文件：** `scripts/train_flow_diffusion.py:46-48`

```python
# 加载配置
config = OmegaConf.load(args.config)
# 实例化数据模块
data = instantiate_from_config(config.data)
# 等价于：data = MNISTLoader(batch_size=256, num_workers=4)
```

### 3. 数据加载器实现

**文件：** `sgm/data/mnist.py:20-72`

```python
class MNISTLoader(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=0, ...):
        # 1. 定义数据转换
        transform = transforms.Compose([
            transforms.ToTensor(),           # [0,1]
            transforms.Lambda(lambda x: x * 2.0 - 1.0)  # [-1,1]
        ])

        # 2. 加载数据集
        self.train_dataset = MNISTDataDictWrapper(
            torchvision.datasets.MNIST(
                root=".data/",       # 保存目录
                train=True,          # 训练集
                download=True,       # 自动下载
                transform=transform
            )
        )

    def train_dataloader(self):
        # 3. 返回 PyTorch DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
```

### 4. 数据格式包装

**文件：** `sgm/data/mnist.py:7-17`

```python
class MNISTDataDictWrapper(Dataset):
    def __getitem__(self, i):
        x, y = self.dset[i]  # 原始：(image, label)
        return {"jpg": x, "cls": y}  # 转换为字典格式
```

**为什么需要包装？** 模型期望的输入格式是字典：
- `"jpg"`: 图像数据（即使不是 jpg 格式）
- `"cls"`: 类别标签（可选，用于条件生成）

### 5. 训练循环使用

**文件：** `scripts/train_flow_diffusion.py:91`

```python
trainer.fit(model, data)
```

在每个训练步骤中：
```python
# sgm/models/score_flow.py:264
def training_step(self, batch, batch_idx):
    x = batch[self.input_key]  # self.input_key = "jpg"
    # batch = {"jpg": Tensor[B, C, H, W], "cls": Tensor[B]}
```

---

## 🔄 替换数据集的三种方法

### 方法 1：使用已有的数据加载器（最简单）

项目已内置：
- `sgm.data.mnist.MNISTLoader` - MNIST
- `sgm.data.cifar10.CIFAR10Loader` - CIFAR-10

**修改配置文件：**
```yaml
data:
  target: sgm.data.cifar10.CIFAR10Loader  # 改这里
  params:
    batch_size: 128
    num_workers: 4
```

### 方法 2：创建新的数据加载器（推荐）

**步骤：**

#### ① 创建文件 `sgm/data/fashion_mnist.py`

```python
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class FashionMNISTDataDictWrapper(Dataset):
    def __init__(self, dset):
        super().__init__()
        self.dset = dset

    def __getitem__(self, i):
        x, y = self.dset[i]
        return {"jpg": x, "cls": y}

    def __len__(self):
        return len(self.dset)

class FashionMNISTLoader(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers=0, shuffle=True):
        super().__init__()

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)  # [-1, 1]
        ])

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.train_dataset = FashionMNISTDataDictWrapper(
            torchvision.datasets.FashionMNIST(
                root=".data/",
                train=True,
                download=True,
                transform=transform
            )
        )

        self.test_dataset = FashionMNISTDataDictWrapper(
            torchvision.datasets.FashionMNIST(
                root=".data/",
                train=False,
                download=True,
                transform=transform
            )
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return self.test_dataloader()
```

#### ② 修改配置文件

```yaml
data:
  target: sgm.data.fashion_mnist.FashionMNISTLoader
  params:
    batch_size: 256
    num_workers: 4
```

### 方法 3：使用 HuggingFace 数据集

**步骤：**

#### ① 创建文件 `sgm/data/huggingface_loader.py`

```python
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchvision import transforms
from PIL import Image

class HuggingFaceDataDictWrapper(Dataset):
    def __init__(self, hf_dataset, transform=None):
        super().__init__()
        self.dataset = hf_dataset
        self.transform = transform

    def __getitem__(self, i):
        item = self.dataset[i]
        # 假设数据集格式：{"image": PIL.Image, "label": int}
        image = item["image"]

        # 确保是 PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # 转换为 RGB（如果是灰度图）
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "jpg": image,
            "cls": item.get("label", 0)  # 如果没有标签，默认为 0
        }

    def __len__(self):
        return len(self.dataset)

class HuggingFaceLoader(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,        # 例如 "cifar10"
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 32,
        dataset_config: str = None,  # 可选子配置
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 定义转换
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0)  # [-1, 1]
        ])

        # 加载 HuggingFace 数据集
        dataset = load_dataset(dataset_name, dataset_config)

        self.train_dataset = HuggingFaceDataDictWrapper(
            dataset["train"], transform=transform
        )
        self.test_dataset = HuggingFaceDataDictWrapper(
            dataset["test"] if "test" in dataset else dataset["validation"],
            transform=transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return self.test_dataloader()
```

#### ② 修改配置文件

```yaml
data:
  target: sgm.data.huggingface_loader.HuggingFaceLoader
  params:
    dataset_name: "fashion_mnist"  # 或 "cifar10", "food101" 等
    batch_size: 128
    num_workers: 4
    image_size: 28  # MNIST/FashionMNIST 用 28，CIFAR 用 32
```

---

## 📋 关键要点

### ✅ 数据格式要求

返回的字典必须包含：
```python
{
    "jpg": Tensor,  # Shape: [C, H, W]，值域：[-1, 1]
    "cls": int,     # 类别标签（可选，用于条件生成）
}
```

### ✅ 数据归一化

**必须归一化到 [-1, 1]：**
```python
transforms.Lambda(lambda x: x * 2.0 - 1.0)  # [0,1] → [-1,1]
```

### ✅ 继承关系

```python
class YourLoader(pl.LightningDataModule):  # 必须继承这个
    def train_dataloader(self):  # 必须实现
        return DataLoader(...)

    def val_dataloader(self):    # 可选
        return DataLoader(...)
```

### ✅ 自动下载

**TorchVision 数据集：** `download=True` 会自动下载

**HuggingFace 数据集：** `load_dataset()` 会自动下载并缓存到 `~/.cache/huggingface/`

---

## 🎯 快速测试数据加载器

```python
# 测试脚本
from sgm.data.mnist import MNISTLoader

loader = MNISTLoader(batch_size=4, num_workers=0)
train_dl = loader.train_dataloader()

# 获取一个 batch
batch = next(iter(train_dl))
print("Keys:", batch.keys())          # dict_keys(['jpg', 'cls'])
print("Image shape:", batch["jpg"].shape)  # torch.Size([4, 1, 28, 28])
print("Labels:", batch["cls"])        # tensor([5, 0, 4, 1])
print("Value range:", batch["jpg"].min(), batch["jpg"].max())  # -1.0, 1.0
```

---

## 📦 常用数据集快速替换

| 数据集 | 配置修改 | 说明 |
|--------|----------|------|
| **MNIST** | `sgm.data.mnist.MNISTLoader` | 已内置，28×28 灰度 |
| **CIFAR-10** | `sgm.data.cifar10.CIFAR10Loader` | 已内置，32×32 彩色 |
| **Fashion-MNIST** | 创建新 loader（方法 2） | 需要自己实现 |
| **HuggingFace** | 使用方法 3 的通用 loader | 支持所有 HF 数据集 |

---

## 🚀 实战示例

### 切换到 CIFAR-10

1. 修改配置：
```yaml
# configs/training/flow_diffusion_cifar.yaml
data:
  target: sgm.data.cifar10.CIFAR10Loader
  params:
    batch_size: 128
    num_workers: 4
```

2. 训练：
```bash
python scripts/train_flow_diffusion.py \
    --config configs/training/flow_diffusion_cifar.yaml \
    --name cifar10_test
```

### 使用 HuggingFace 数据集

1. 先实现方法 3 的 `HuggingFaceLoader`
2. 修改配置：
```yaml
data:
  target: sgm.data.huggingface_loader.HuggingFaceLoader
  params:
    dataset_name: "food101"
    batch_size: 64
    num_workers: 4
    image_size: 64
```

---

**总结：** 数据集替换只需修改配置文件中的 `target` 和 `params`！
