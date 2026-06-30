# Vision Transformer (ViT)

## 为什么需要 ViT？

在 ViT 之前，计算机视觉领域几乎被 **CNN**（卷积神经网络）统治：

- ResNet、VGG、Inception 等模型
- 卷积操作提取局部特征
- 池化操作降低分辨率

**问题**：CNN 只能捕捉局部信息，难以捕捉全局依赖关系。

**ViT**（Vision Transformer）将 Transformer 应用于视觉任务，打破了 CNN 的垄断。

---

## 核心思想

### 从 NLP 到 CV

Transformer 在 NLP 中取得了巨大成功：

```
文本 → 分词 → Token → Transformer → 输出
```

ViT 的核心思想：

> 将图像分割成 patches，每个 patch 视为一个 token，然后用 Transformer 处理。

### 流程

```
输入图像 → 分割成 Patches → 线性嵌入 → 位置编码 → Transformer → 分类
```

---

## Patch Embedding

### 为什么需要 Patch Embedding？

Transformer 的输入是序列，但图像是 2D 的。我们需要将图像转换为序列。

### 分割成 Patches

将图像分割成固定大小的 patches：

- 输入图像：224 × 224 × 3
- Patch 大小：16 × 16
- Patches 数量：(224/16) × (224/16) = 14 × 14 = 196 个 patches

每个 patch 是 16 × 16 × 3 = 768 维的向量。

### 代码实现

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        """
        Patch Embedding

        将图像分割成 patches，并映射到嵌入空间

        参数:
            img_size: 图像大小
            patch_size: patch 大小
            in_channels: 输入通道数（RGB=3）
            embed_dim: 嵌入维度
        """
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # 用卷积实现 patch 分割和线性映射
        # kernel_size = stride = patch_size，实现不重叠的 patch 分割
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        前向传播

        输入: (B, C, H, W)
        输出: (B, num_patches, embed_dim)
        """
        # 卷积: (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.projection(x)

        # 展平: (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = x.flatten(2)

        # 转置: (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)

        return x
```

### 直观理解

```
输入图像 (224×224×3)
        ↓
分割成 14×14 = 196 个 patches
        ↓
每个 patch (16×16×3) 展平为 768 维向量
        ↓
输出: 196 个 768 维的 token
```

---

## 位置编码

### 为什么需要位置编码？

Transformer 是**置换不变**的（permutation invariant），即打乱输入顺序不影响输出。

但图像的 patch 顺序很重要（左上角的 patch 和右下角的 patch 含义不同），所以我们需要添加位置信息。

### 位置编码类型

1. **可学习的位置编码**：每个位置一个可学习的向量
2. **正弦位置编码**：使用正弦函数生成

ViT 使用**可学习的位置编码**：

```python
# 位置编码：(1, num_patches + 1, embed_dim)
# +1 是因为还有 CLS token
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
```

### 代码实现

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS Token：用于分类的特殊 token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码：可学习
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer 编码器
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        # 层归一化
        self.norm = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # 1. Patch Embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # 2. 添加 CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)

        # 3. 添加位置编码
        x = x + self.pos_embed  # (B, num_patches + 1, embed_dim)

        # 4. Transformer 编码器
        x = self.blocks(x)  # (B, num_patches + 1, embed_dim)

        # 5. 层归一化
        x = self.norm(x)  # (B, num_patches + 1, embed_dim)

        # 6. 分类：使用 CLS Token 的输出
        return self.head(x[:, 0])  # (B, num_classes)
```

---

## CLS Token

### 什么是 CLS Token？

CLS（Classification）Token 是一个**特殊的 token**，添加到 patch 序列的开头。

```
[CLS] patch_1 patch_2 ... patch_196
```

### 为什么需要 CLS Token？

1. **统一表示**：CLS Token 会聚合所有 patch 的信息
2. **分类任务**：用 CLS Token 的输出进行分类
3. **NLP 借鉴**：BERT 中也有 CLS Token

### 代码实现

```python
# CLS Token：(1, 1, embed_dim)
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

# 前向传播中
cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
```

---

## 多头自注意力

### 核心公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中：
- $Q$：查询矩阵
- $K$：键矩阵
- $V$：值矩阵
- $d_k$：键的维度

### 代码实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        """
        多头自注意力

        参数:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV 投影
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)

        # 输出投影
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape

        # 1. 计算 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2. 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 3. 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 4. 输出投影
        return self.proj(x)
```

---

## 训练配置

### 超参数

```python
config = {
    'img_size': 224,        # 图像大小
    'patch_size': 16,       # patch 大小
    'num_classes': 1000,    # 类别数
    'embed_dim': 768,       # 嵌入维度
    'depth': 12,            # Transformer 层数
    'num_heads': 12,        # 注意力头数
    'batch_size': 256,      # 批次大小
    'lr': 1e-3,             # 学习率
    'epochs': 300,          # 训练轮数
}
```

### 数据增强

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## ViT vs CNN

### 优势

| 特性 | CNN | ViT |
|------|-----|-----|
| 全局信息 | 弱（局部感受野） | 强（全局注意力） |
| 位置信息 | 隐式（卷积核） | 显式（位置编码） |
| 可扩展性 | 中 | 强 |
| 预训练效果 | 好 | 更好 |

### 劣势

| 特性 | CNN | ViT |
|------|-----|-----|
| 数据需求 | 少 | 多（需要大规模预训练） |
| 计算开销 | 小 | 大（自注意力复杂度高） |
| 归纳偏置 | 强（平移不变性） | 弱 |

---

## 深入讨论

### 为什么 ViT 需要大量数据？

CNN 有**归纳偏置**（inductive bias）：
- **平移不变性**：卷积核在整个图像上滑动
- **局部性**：卷积核只看局部区域

ViT 没有这些归纳偏置，所以需要更多数据来学习这些特性。

### 数据不足时怎么办？

1. **使用预训练模型**：在 ImageNet 上预训练
2. **数据增强**：增加训练数据多样性
3. **知识蒸馏**：从 CNN 教师模型学习

### Patch 大小的影响

- **Patch 越小**：序列越长，计算开销越大，但能捕捉更细粒度的信息
- **Patch 越大**：序列越短，计算开销小，但可能丢失细节

---

## 关键要点

1. **ViT 将图像分割成 patches**，每个 patch 视为一个 token
2. **Patch Embedding** 用卷积实现 patch 分割和线性映射
3. **位置编码** 添加位置信息，弥补 Transformer 的置换不变性
4. **CLS Token** 聚合所有 patch 的信息，用于分类
5. **需要大量数据** 预训练，数据不足时效果不如 CNN

---

## 延伸阅读

1. **DeiT**：数据高效的 ViT，使用知识蒸馏
2. **Swin Transformer**：层次化的 ViT，使用滑动窗口注意力
3. **BEiT**：自监督预训练的 ViT
