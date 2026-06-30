# MicroGPT 实现详解

## 1.1 数据集

### 数据集准备
- 文本数据的收集和预处理
- 数据清洗和格式化
- 训练集/验证集划分

### 数据格式
```python
# 输入-目标对示例
输入: "The cat sat on the"
目标: "mat"
```

---

## 1.2 分词器 (Tokenizer)

### 什么是分词器
分词器是将文本转换为数字序列的工具。

### 常见分词方法
1. **字符级 (Character-level)**: 每个字符作为一个token
2. **词级 (Word-level)**: 每个词作为一个token
3. **子词级 (Subword-level)**: BPE, WordPiece等

### BPE算法
```python
# 简化版BPE
def bpe_tokenize(text, vocab_size):
    # 1. 初始化词表为所有字符
    vocab = set(text)
    # 2. 迭代合并最频繁的pair
    while len(vocab) < vocab_size:
        pairs = count_pairs(text)
        best_pair = max(pairs, key=pairs.get)
        vocab.add(best_pair)
        text = merge_pair(text, best_pair)
    return vocab
```

---

## 1.3 自动微分 (Autograd)

### PyTorch Autograd 核心概念

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
- **前向传播**: 构建计算图
- **反向传播**: 计算梯度
- **梯度清零**: 每次迭代前需要清零

```python
optimizer.zero_grad()  # 清零梯度
loss.backward()        # 反向传播
optimizer.step()       # 更新参数
```

---

## 1.4 参数 (Parameters)

### 模型参数初始化

```python
import torch.nn as nn

# 常用初始化方法
nn.init.xavier_uniform_(tensor)  # Xavier均匀分布
nn.init.kaiming_normal_(tensor)  # Kaiming正态分布
nn.init.zeros_(tensor)           # 全零初始化
```

### 参数数量计算
```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

---

## 1.5 架构 (Architecture)

### Transformer 核心组件

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(output)
```

### 前馈网络
```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))
```

---

## 1.6 训练循环

### 基本训练循环

```python
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()          # 清零梯度
        output = model(data)           # 前向传播
        loss = criterion(output, target)  # 计算损失
        loss.backward()                # 反向传播
        optimizer.step()               # 更新参数

        total_loss += loss.item()

    return total_loss / len(train_loader)
```

### 学习率调度
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scheduler.step()  # 每个epoch后调用
```

---

## 1.7 推理 (Inference)

### 文本生成

```python
def generate(model, start_text, max_length=100, temperature=1.0):
    model.eval()
    tokens = tokenizer.encode(start_text)

    with torch.no_grad():
        for _ in range(max_length):
            # 获取模型输出
            output = model(torch.tensor([tokens]))
            logits = output[:, -1, :] / temperature

            # 采样下一个token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            tokens.append(next_token)

            # 检查结束条件
            if next_token == tokenizer.eos_token:
                break

    return tokenizer.decode(tokens)
```

---

## 1.8 运行

### 完整训练流程

```python
# 1. 准备数据
train_loader = prepare_data()

# 2. 创建模型
model = MicroGPT(vocab_size, d_model, n_heads, n_layers)

# 3. 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# 4. 训练
for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch}, Loss: {loss}")

    # 5. 生成示例
    if epoch % 10 == 0:
        print(generate(model, "The"))
```

---

## 1.9 真实世界

### 实际应用考虑

1. **数据质量**: 高质量数据比大量低质量数据更重要
2. **计算资源**: GPU/TPU的使用
3. **分布式训练**: 多GPU/多机训练
4. **模型保存**: 定期保存检查点

```python
# 保存模型
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# 加载模型
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

---

## 1.10 FAQ

### 常见问题

**Q: 为什么loss不下降？**
A: 检查学习率、数据预处理、模型架构

**Q: 如何选择超参数？**
A: 从标准配置开始，逐步调整

**Q: 过拟合怎么办？**
A: 增加数据、使用dropout、early stopping

---

## 关键要点总结

1. **数据预处理**: 分词、编码、批处理
2. **模型架构**: Transformer的核心组件
3. **训练技巧**: 学习率调度、梯度裁剪
4. **推理策略**: Temperature采样、Top-k/Top-p采样

---

*参考: 原文档 Chapter 1*
