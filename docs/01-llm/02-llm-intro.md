# LLM 简介

## 大语言模型要解决的问题

### 核心任务

给定一段文本，预测下一个最可能的词（token）。

```
输入: "The cat sat on the"
输出: "mat" (概率最高)
```

---

## 如何训练预测下一个 token 的模型？

### 自回归语言模型

$$P(x_1, x_2, ..., x_n) = \prod_{i=1}^{n} P(x_i | x_1, ..., x_{i-1})$$

### 训练目标

最大化下一个 token 的对数似然：

$$L = \sum_{i=1}^{n} \log P(x_i | x_1, ..., x_{i-1})$$

---

## 训练数据格式

### 输入-目标对

```python
# 原始文本
text = "The cat sat on the mat"

# 创建训练样本
# 输入: "The cat" -> 目标: " sat"
# 输入: "The cat sat" -> 目标: " on"
# 输入: "The cat sat on" -> 目标: " the"
# 输入: "The cat sat on the" -> 目标: " mat"
```

### 数据组织

```python
def create_training_data(text, seq_length):
    tokens = tokenizer.encode(text)
    inputs = []
    targets = []

    for i in range(len(tokens) - seq_length):
        inputs.append(tokens[i:i + seq_length])
        targets.append(tokens[i + 1:i + seq_length + 1])

    return inputs, targets
```

---

## 模型结构设计

### Transformer 架构

```
输入 Token
    ↓
词嵌入 (Token Embedding)
    ↓
位置编码 (Positional Encoding)
    ↓
┌─────────────────┐
│  Transformer    │
│  Block × N      │
│  - 注意力层     │
│  - 前馈网络     │
│  - 层归一化     │
└─────────────────┘
    ↓
输出层 (Linear + Softmax)
    ↓
预测的下一个 Token
```

---

## 损失函数设计

### 交叉熵损失

```python
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# logits: 模型输出 (batch_size, seq_len, vocab_size)
# targets: 目标 token (batch_size, seq_len)
loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
```

### 困惑度 (Perplexity)

$$PPL = \exp(L)$$

困惑度越低，模型越好。

---

## 关键要点

1. **LLM 本质是分类模型**：预测下一个 token 的概率分布
2. **自回归**：使用前面的 token 预测后面的 token
3. **Transformer**：目前最成功的架构
4. **交叉熵损失**：标准的分类损失函数
