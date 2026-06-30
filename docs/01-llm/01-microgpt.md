# MicroGPT 实现详解

## 概述

MicroGPT 是一个简化版的 GPT 模型实现，帮助理解大语言模型的核心原理。

---

## 数据集

### 数据准备

```python
# 文本数据示例
text = "The cat sat on the mat. The dog ran in the park..."

# 分词
tokens = tokenizer.encode(text)

# 创建输入-目标对
# 输入: "The cat sat on the"
# 目标: "mat"
```

---

## 分词器 (Tokenizer)

### BPE 算法

```python
class SimpleTokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}

    def train(self, text):
        # 初始化词表为所有字符
        chars = sorted(list(set(text)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}

        # BPE 合并
        while len(self.vocab) < self.vocab_size:
            pairs = self._count_pairs(text)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self._merge_pair(best_pair, text)

    def encode(self, text):
        # 将文本转换为 token ID
        return [self.vocab.get(ch, 0) for ch in text]

    def decode(self, ids):
        # 将 token ID 转换回文本
        return ''.join([self.vocab.get(i, '') for i in ids])
```

---

## 自动微分 (Autograd)

### PyTorch Autograd

```python
import torch

# 创建需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

# 自动计算梯度
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7
```

### 计算图

```python
# 前向传播构建计算图
a = x + 2
b = a * 3
c = b ** 2

# 反向传播计算梯度
c.backward()
```

---

## 模型架构

### Transformer 块

```python
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 自注意力 + 残差连接
        attn_out = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # 前馈网络 + 残差连接
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x
```

### MicroGPT 模型

```python
class MicroGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_model * 4)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.fc_out(x)
```

---

## 训练循环

```python
def train(model, train_loader, optimizer, criterion, epochs=10):
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")
```

---

## 文本生成

```python
def generate(model, start_text, max_length=100, temperature=1.0):
    model.eval()
    tokens = tokenizer.encode(start_text)

    with torch.no_grad():
        for _ in range(max_length):
            output = model(torch.tensor([tokens]))
            logits = output[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)

            if next_token == tokenizer.eos_token:
                break

    return tokenizer.decode(tokens)
```

---

## 关键要点

1. **分词器**将文本转换为数字序列
2. **Transformer** 是 LLM 的核心架构
3. **自注意力**机制捕捉序列中的依赖关系
4. **训练**通过预测下一个 token 来学习语言模式
