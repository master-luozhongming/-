# 直接偏好优化 (DPO)

## 为什么需要 DPO？

传统的 RLHF（基于人类反馈的强化学习）流程：

1. **训练奖励模型**：从人类偏好数据中学习奖励函数
2. **强化学习优化**：用 PPO 优化策略，最大化奖励

这个流程有两个问题：

1. **奖励模型可能不准确**：人类偏好数据有限，奖励模型可能过拟合
2. **PPO 训练不稳定**：需要调参，容易崩溃

**DPO**（Direct Preference Optimization）直接从偏好数据优化策略，**不需要训练奖励模型**。

---

## 核心思想

### 传统 RLHF 的目标

$$\max_{\pi_\theta} E_{x \sim D, y \sim \pi_\theta(y|x)}[R(x, y)] - \beta \cdot KL(\pi_\theta || \pi_{ref})$$

其中：
- $R(x, y)$：奖励函数
- $\pi_{ref}$：参考策略（通常是 SFT 模型）
- $\beta$：KL 惩罚系数

### 最优策略的闭式解

这个优化问题有闭式解：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{R(x, y)}{\beta}\right)$$

其中 $Z(x) = \sum_y \pi_{ref}(y|x) \exp\left(\frac{R(x, y)}{\beta}\right)$ 是归一化常数。

### 关键洞察

从最优策略公式反解出奖励函数：

$$R(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**这个公式的意义**：奖励函数可以用策略和参考策略的比率来表示！

---

## 从奖励到偏好

### Bradley-Terry 模型

人类偏好可以用 Bradley-Terry 模型表示：

$$P(y_w \succ y_l | x) = \sigma(R(x, y_w) - R(x, y_l))$$

其中：
- $y_w$：偏好的响应（winner）
- $y_l$：非偏好的响应（loser）
- $\sigma$：sigmoid 函数

### 代入奖励公式

把 $R(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$ 代入：

$$P(y_w \succ y_l | x) = \sigma\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

注意：$\beta \log Z(x)$ 在相减时消掉了！

---

## DPO 损失函数

### 推导

我们想最大化偏好数据的似然：

$$\max_\theta \sum_{(x, y_w, y_l)} \log P(y_w \succ y_l | x)$$

代入 Bradley-Terry 模型：

$$\max_\theta \sum_{(x, y_w, y_l)} \log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

### 最终公式

DPO 损失函数：

$$L(\theta) = -E_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

### 直观理解

- 如果 $\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} > \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$：
  - $\sigma(...)$ 接近 1
  - $-\log \sigma(...)$ 接近 0
  - 损失小

- 如果 $\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} < \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}$：
  - $\sigma(...)$ 接近 0
  - $-\log \sigma(...)$ 接近 ∞
  - 损失大

**效果**：让策略更倾向于生成偏好响应，而不是非偏好响应。

---

## 代码实现

### 计算 log 概率

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_logprobs(model, input_ids, labels):
    """
    计算序列的 log 概率

    log π(y|x) = ∑_t log π(y_t | y_{<t}, x)
    """
    outputs = model(input_ids=input_ids, labels=labels)
    logits = outputs.logits[:, :-1, :]  # 去掉最后一个位置
    labels = labels[:, 1:]  # 去掉第一个位置

    # 计算 log softmax
    log_probs = torch.log_softmax(logits, dim=-1)

    # 提取每个 token 的 log 概率
    token_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)

    # 返回序列的平均 log 概率
    return token_log_probs.mean(dim=-1)
```

### DPO 损失函数

```python
def dpo_loss(model, ref_model, tokenizer, prompt, chosen, rejected, beta=0.1):
    """
    DPO 损失函数

    L(θ) = -E[log σ(β log π_θ(y_w|x)/π_ref(y_w|x) - β log π_θ(y_l|x)/π_ref(y_l|x))]
    """
    # 编码输入
    chosen_inputs = tokenizer(prompt + chosen, return_tensors="pt")
    rejected_inputs = tokenizer(prompt + rejected, return_tensors="pt")

    # 当前策略的 log 概率
    chosen_logprobs = compute_logprobs(
        model, chosen_inputs["input_ids"], chosen_inputs["input_ids"]
    )
    rejected_logprobs = compute_logprobs(
        model, rejected_inputs["input_ids"], rejected_inputs["input_ids"]
    )

    # 参考策略的 log 概率
    with torch.no_grad():
        ref_chosen_logprobs = compute_logprobs(
            ref_model, chosen_inputs["input_ids"], chosen_inputs["input_ids"]
        )
        ref_rejected_logprobs = compute_logprobs(
            ref_model, rejected_inputs["input_ids"], rejected_inputs["input_ids"]
        )

    # 计算 log 比率
    chosen_logratios = chosen_logprobs - ref_chosen_logprobs
    rejected_logratios = rejected_logprobs - ref_rejected_logprobs

    # DPO 损失
    logits = beta * (chosen_logratios - rejected_logratios)
    loss = -torch.log(torch.sigmoid(logits)).mean()

    return loss
```

### 训练循环

```python
def train_dpo():
    """DPO 训练流程"""
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

    # 训练数据
    data = [
        {
            "prompt": "什么是机器学习？",
            "chosen": "机器学习是人工智能的一个分支，它让计算机能够从数据中学习，而不需要显式编程。",
            "rejected": "机器学习就是让机器自己学习。"
        }
    ]

    # 训练循环
    for epoch in range(10):
        for batch in data:
            # 计算损失
            loss = dpo_loss(
                model, ref_model, tokenizer,
                batch["prompt"], batch["chosen"], batch["rejected"]
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## DPO vs PPO

### 优势

| 特性 | PPO | DPO |
|------|-----|-----|
| 奖励模型 | 需要 | 不需要 |
| 训练复杂度 | 高 | 低 |
| 数据需求 | 在线采样 | 离线偏好数据 |
| 稳定性 | 需要调参 | 更稳定 |
| 实现难度 | 难 | 容易 |

### 劣势

| 特性 | PPO | DPO |
|------|-----|-----|
| 探索能力 | 强 | 弱 |
| 适用场景 | 通用 | 有偏好数据时 |
| 理论保证 | 更强 | 较弱 |

---

## 深入讨论

### 为什么 DPO 更稳定？

1. **没有奖励模型误差传播**：直接从偏好数据学习，不需要训练奖励模型
2. **没有在线采样**：使用离线数据，不需要与环境交互
3. **损失函数简单**：只有交叉熵损失，没有复杂的策略梯度

### $\beta$ 的作用

- $\beta$ 控制策略与参考策略的差异
- $\beta$ 越大：策略越接近参考策略
- $\beta$ 越小：策略越远离参考策略

### 什么时候用 DPO？

1. **有高质量偏好数据**：人类标注的偏好对
2. **计算资源有限**：不想训练奖励模型
3. **追求稳定性**：不想花时间调参

### 什么时候用 PPO？

1. **没有偏好数据**：需要在线采样
2. **需要探索**：任务复杂，需要探索
3. **追求最优性能**：愿意花时间调参

---

## 关键要点

1. **DPO 直接从偏好数据优化策略**，不需要奖励模型
2. **核心公式**：$L(\theta) = -E[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$
3. **更稳定、更简单**，但探索能力较弱
4. **适用于**有高质量偏好数据的场景

---

## 延伸阅读

1. **RLHF**：传统的基于人类反馈的强化学习
2. **IPO**：Identity Preference Optimization，DPO 的改进版本
3. **KTO**：Kahneman-Tversky Optimization，不需要成对偏好数据
