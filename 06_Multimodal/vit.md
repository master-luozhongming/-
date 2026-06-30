# Vision Transformer (ViT)

## 概述

Vision Transformer (ViT) 是将 Transformer 架构应用于计算机视觉任务的模型，打破了 CNN 在视觉领域的垄断。

## 核心思想

将图像分割成小块 (patches)，将每个 patch 视为一个 token，然后使用标准 Transformer 进行处理。

## 架构

```
输入图像 → 分割成 Patches → 线性嵌入 → 位置编码 → Transformer 编码器 → 分类头
```

### 详细流程

1. **Patch Embedding**: 将图像分割成固定大小的 patches
2. **线性投影**: 将每个 patch 投影到 D 维向量
3. **位置编码**: 添加位置信息
4. **[CLS] Token**: 添加分类 token
5. **Transformer 编码器**: 多层自注意力和前馈网络
6. **分类头**: MLP 用于最终分类

---

## 代码实现

### Patch Embedding

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 使用卷积实现 patch embedding
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            patches: (batch_size, num_patches, embed_dim)
        """
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.projection(x)

        # (B, embed_dim, H/P, W/P) -> (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)

        return x
```

### 多头自注意力

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # 计算 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 应用注意力到 V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x
```

### Transformer 编码器

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
```

### Vision Transformer

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # [CLS] token 和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        # Transformer 编码器
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)

        # 添加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer 编码器
        x = self.blocks(x)

        # 分类
        x = self.norm(x)
        cls_token = x[:, 0]  # 使用 [CLS] token
        x = self.head(cls_token)

        return x
```

---

## 训练配置

```python
def train_vit():
    # 超参数
    config = {
        'img_size': 224,
        'patch_size': 16,
        'num_classes': 1000,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'batch_size': 256,
        'lr': 1e-3,
        'epochs': 300,
        'warmup_epochs': 10,
        'weight_decay': 0.05,
    }

    # 创建模型
    model = VisionTransformer(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_classes=config['num_classes'],
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads']
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )

    return model, optimizer, scheduler
```

---

## 使用 MNIST 数据集

```python
from torchvision import datasets, transforms

def load_mnist():
    transform = transforms.Compose([
        transforms.Resize(224),  # ViT 需要较大的输入
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )

    return train_loader, test_loader
```

---

## ViT 变体

| 模型 | 层数 | 隐藏维度 | 注意力头 | 参数量 |
|------|------|----------|----------|--------|
| ViT-Small | 12 | 384 | 6 | 22M |
| ViT-Base | 12 | 768 | 12 | 86M |
| ViT-Large | 24 | 1024 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 16 | 632M |

---

## 与 CNN 的对比

| 特性 | CNN | ViT |
|------|-----|-----|
| 归纳偏置 | 局部性、平移不变性 | 较少 |
| 数据效率 | 高（小数据集） | 低（需要大数据集） |
| 计算效率 | 高 | 中 |
| 可扩展性 | 中 | 高 |

---

## 关键要点

1. **ViT 将 Transformer 应用于视觉任务**
2. **将图像分割成 patches 作为 token**
3. **需要大量数据进行预训练**
4. **在大规模数据集上优于 CNN**

---

*参考: 原文档 Chapter 14*
