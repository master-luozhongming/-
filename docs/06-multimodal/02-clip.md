# CLIP 模型

## 概述

CLIP (Contrastive Language-Image Pre-training) 通过对比学习连接图像和文本。

---

## 核心思想

```
图像 ─────► 图像编码器 ─────► 图像特征
                                    ↓
                              对比学习
                                    ↑
文本 ─────► 文本编码器 ─────► 文本特征
```

---

## 对比学习

### InfoNCE 损失

```python
def contrastive_loss(image_features, text_features, temperature=0.07):
    # 归一化
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # 计算相似度
    logits = image_features @ text_features.T / temperature

    # 对角线是正样本
    labels = torch.arange(len(logits), device=logits.device)

    # 双向对比损失
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2
```

---

## 图像编码器

```python
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        # 使用 ViT 或 ResNet
        self.vit = VisionTransformer(embed_dim=embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, images):
        features = self.vit(images)
        return self.proj(features)
```

---

## 文本编码器

```python
class TextEncoder(nn.Module):
    def __init__(self, vocab_size=49408, embed_dim=512, max_length=77):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_length, embed_dim)
        self.transformer = TransformerEncoder(embed_dim, 8, 4)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, text):
        B, T = text.shape
        positions = torch.arange(T, device=text.device)

        x = self.token_embed(text) + self.pos_embed(positions)
        x = self.transformer(x)
        x = x[torch.arange(B), text.argmax(dim=-1)]

        return self.proj(x)
```

---

## CLIP 模型

```python
class CLIP(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, images, text):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(text)

        return contrastive_loss(image_features, text_features, self.temperature)

    def encode_image(self, images):
        return self.image_encoder(images)

    def encode_text(self, text):
        return self.text_encoder(text)
```

---

## 零样本分类

```python
def zero_shot_classify(model, image, class_names):
    # 编码图像
    image_features = model.encode_image(image)

    # 创建文本提示
    text_inputs = [f"a photo of a {name}" for name in class_names]
    text_features = model.encode_text(tokenizer(text_inputs))

    # 计算相似度
    similarity = image_features @ text_features.T

    return class_names[similarity.argmax()]
```

---

## 关键要点

1. **对比学习**连接图像和文本
2. **零样本分类**无需微调
3. **多模态表示**统一的特征空间
4. **广泛应用**图像检索、图像生成等
