# CLIP 模型

## 为什么需要 CLIP？

传统的计算机视觉模型有一个问题：**只能识别训练过的类别**。

比如，一个在 ImageNet（1000 类）上训练的模型：
- ✅ 可以识别"猫"、"狗"、"汽车"等
- ❌ 无法识别"柯基"、"哈士奇"等（如果没见过）

**CLIP**（Contrastive Language-Image Pre-training）通过**对比学习**连接图像和文本，实现**零样本分类**。

---

## 核心思想

### 从文本到图像

CLIP 的核心思想：

> 用文本描述图像，让模型学会图像和文本的对应关系。

### 流程

```
图像 ─────► 图像编码器 ─────► 图像特征
                                    ↓
                              对比学习
                                    ↑
文本 ─────► 文本编码器 ─────► 文本特征
```

### 对比学习

对比学习的目标：

- **正样本**：图像和对应的文本描述
- **负样本**：图像和不对应的文本描述

让正样本的相似度**高**，负样本的相似度**低**。

---

## InfoNCE 损失

### 公式

InfoNCE 损失：

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}$$

其中：
- $I_i$：第 $i$ 个图像的特征
- $T_i$：第 $i$ 个文本的特征
- $\text{sim}(I, T)$：余弦相似度
- $\tau$：温度参数

### 直观理解

对于第 $i$ 个图像：
- 分子：$\exp(\text{sim}(I_i, T_i) / \tau)$，正样本的相似度
- 分母：$\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)$，所有样本的相似度之和

**目标**：让正样本的相似度占总相似度的比例最大化。

### 代码实现

```python
import torch
import torch.nn.functional as F

def contrastive_loss(image_features, text_features, temperature=0.07):
    """
    InfoNCE 对比损失

    L = -1/N ∑_i log exp(sim(I_i, T_i)/τ) / ∑_j exp(sim(I_i, T_j)/τ)
    """
    # 1. 归一化特征
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # 2. 计算相似度矩阵
    # (N, D) @ (D, N) = (N, N)
    logits = image_features @ text_features.T / temperature

    # 3. 创建标签：对角线是正样本
    labels = torch.arange(len(logits), device=logits.device)

    # 4. 双向对比损失
    # 图像 → 文本
    loss_i2t = F.cross_entropy(logits, labels)
    # 文本 → 图像
    loss_t2i = F.cross_entropy(logits.T, labels)

    # 5. 平均
    return (loss_i2t + loss_t2i) / 2
```

### 为什么是双向对比？

- **图像 → 文本**：给定图像，找到对应的文本
- **文本 → 图像**：给定文本，找到对应的图像

双向对比可以学习更全面的表示。

---

## 图像编码器

### 架构

CLIP 的图像编码器可以使用：

1. **ResNet**：CNN 架构
2. **ViT**：Transformer 架构

这里以 ViT 为例：

```python
import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        """
        图像编码器

        使用 ViT 提取图像特征
        """
        super().__init__()

        # 使用 ViT
        self.vit = VisionTransformer(embed_dim=embed_dim)

        # 投影层：将特征映射到共享空间
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, images):
        """
        前向传播

        输入: (B, C, H, W)
        输出: (B, embed_dim)
        """
        # 提取特征
        features = self.vit(images)  # (B, embed_dim)

        # 投影到共享空间
        return self.proj(features)  # (B, embed_dim)
```

---

## 文本编码器

### 架构

CLIP 的文本编码器使用 Transformer：

```python
class TextEncoder(nn.Module):
    def __init__(self, vocab_size=49408, embed_dim=512, max_length=77):
        """
        文本编码器

        使用 Transformer 提取文本特征
        """
        super().__init__()

        # Token 嵌入
        self.token_embed = nn.Embedding(vocab_size, embed_dim)

        # 位置嵌入
        self.pos_embed = nn.Embedding(max_length, embed_dim)

        # Transformer 编码器
        self.transformer = TransformerEncoder(embed_dim, 8, 4)

        # 投影层
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, text):
        """
        前向传播

        输入: (B, T)
        输出: (B, embed_dim)
        """
        B, T = text.shape

        # 1. Token 嵌入 + 位置嵌入
        positions = torch.arange(T, device=text.device)
        x = self.token_embed(text) + self.pos_embed(positions)

        # 2. Transformer 编码
        x = self.transformer(x)

        # 3. 提取 [EOS] token 的特征
        # CLIP 使用 [EOS] token 的特征作为整个句子的表示
        x = x[torch.arange(B), text.argmax(dim=-1)]

        # 4. 投影到共享空间
        return self.proj(x)
```

### 为什么用 [EOS] token？

- [EOS]（End of Sentence）token 位于句子末尾
- 它会聚合整个句子的信息
- 类似于 ViT 中的 CLS token

---

## CLIP 模型

### 完整实现

```python
class CLIP(nn.Module):
    def __init__(self, embed_dim=512):
        """
        CLIP 模型

        连接图像和文本的对比学习模型
        """
        super().__init__()

        # 图像编码器
        self.image_encoder = ImageEncoder(embed_dim)

        # 文本编码器
        self.text_encoder = TextEncoder(embed_dim=embed_dim)

        # 温度参数：可学习
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, images, text):
        """
        前向传播

        计算对比损失
        """
        # 1. 编码图像
        image_features = self.image_encoder(images)

        # 2. 编码文本
        text_features = self.text_encoder(text)

        # 3. 计算对比损失
        return contrastive_loss(image_features, text_features, self.temperature)

    def encode_image(self, images):
        """编码图像"""
        return self.image_encoder(images)

    def encode_text(self, text):
        """编码文本"""
        return self.text_encoder(text)
```

---

## 零样本分类

### 核心思想

CLIP 的强大之处在于**零样本分类**：

> 不需要微调，直接用文本描述来分类图像。

### 流程

1. **准备文本提示**：为每个类别创建文本描述
2. **编码图像和文本**：用 CLIP 编码
3. **计算相似度**：找到最相似的文本
4. **分类**：选择相似度最高的类别

### 代码实现

```python
def zero_shot_classify(model, image, class_names, tokenizer):
    """
    零样本分类

    参数:
        model: CLIP 模型
        image: 输入图像
        class_names: 类别名称列表
        tokenizer: 分词器
    """
    # 1. 编码图像
    image_features = model.encode_image(image)

    # 2. 创建文本提示
    # "a photo of a {class_name}"
    text_inputs = [f"a photo of a {name}" for name in class_names]
    text_features = model.encode_text(tokenizer(text_inputs))

    # 3. 计算相似度
    similarity = image_features @ text_features.T

    # 4. 返回最相似的类别
    return class_names[similarity.argmax()]
```

### 示例

```python
# 类别名称
class_names = ["cat", "dog", "car", "bird"]

# 零样本分类
predicted_class = zero_shot_classify(model, image, class_names, tokenizer)
print(f"Predicted: {predicted_class}")
```

---

## CLIP vs 传统分类

### 优势

| 特性 | 传统分类 | CLIP |
|------|----------|------|
| 类别限制 | 只能识别训练过的类别 | 可以识别任意类别 |
| 数据需求 | 需要大量标注数据 | 不需要标注数据 |
| 泛化能力 | 弱 | 强 |
| 多模态 | 否 | 是 |

### 劣势

| 特性 | 传统分类 | CLIP |
|------|----------|------|
| 精度 | 高（特定任务） | 中（通用） |
| 计算开销 | 小 | 大 |
| 可解释性 | 强 | 弱 |

---

## 深入讨论

### 为什么 CLIP 能实现零样本分类？

1. **共享嵌入空间**：图像和文本映射到同一个空间
2. **语义理解**：文本描述包含了语义信息
3. **对比学习**：学会了图像和文本的对应关系

### 温度参数 $\tau$ 的作用

- $\tau$ 越小：分布越尖锐，模型越自信
- $\tau$ 越大：分布越平滑，模型越不确定
- CLIP 使用可学习的 $\tau$，初始值为 0.07

### CLIP 的局限性

1. **细粒度分类**：难以区分相似类别（如不同品种的狗）
2. **组合推理**：难以理解复杂的空间关系
3. **否定理解**：难以理解否定句

---

## 关键要点

1. **CLIP 通过对比学习连接图像和文本**
2. **InfoNCE 损失**让正样本相似度高，负样本相似度低
3. **零样本分类**不需要微调，直接用文本描述分类
4. **共享嵌入空间**是 CLIP 的核心
5. **广泛应用**：图像检索、图像生成、多模态理解

---

## 延伸阅读

1. **ALIGN**：大规模对比学习的图像-文本模型
2. **BLIP**：统一的视觉-语言预训练框架
3. **Stable Diffusion**：使用 CLIP 作为文本编码器的图像生成模型
