# Vision Transformer (ViT)

## 概述

ViT 将 Transformer 应用于计算机视觉，打破了 CNN 在视觉领域的垄断。

---

## 核心思想

将图像分割成 patches，每个 patch 视为一个 token：

```
输入图像 → 分割成 Patches → 线性嵌入 → 位置编码 → Transformer → 分类
```

---

## Patch Embedding

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x
```

---

## 多头自注意力

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)
```

---

## Vision Transformer

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)

        return self.head(x[:, 0])
```

---

## 训练配置

```python
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
}
```

---

## 关键要点

1. **Patch Embedding** 将图像转换为序列
2. **[CLS] Token** 用于分类
3. **位置编码** 添加位置信息
4. **需要大量数据** 预训练
