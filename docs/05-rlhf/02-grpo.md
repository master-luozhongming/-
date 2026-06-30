# 组相对策略优化 (GRPO)

## 为什么需要 GRPO？

PPO 在 LLM 微调中有一个问题：**需要训练一个 Critic 网络**（价值网络）。

对于大语言模型：
- Critic 网络和 Actor 网络一样大，占用大量显存
- Critic 网络的训练本身就不稳定
- 增加了训练复杂度

**GRPO**（Group Relative Policy Optimization）是专门为 LLM 设计的算法，**不需要 Critic 网络**。

---

## 核心思想

### 组内相对比较

GRPO 的核心思想是：

> 对每个 prompt 采样多个 response，用组内相对优势来替代 Critic 网络。

### 直观理解

想象你在考试：
- **传统 PPO**：需要一个老师（Critic）告诉你每个答案的分数
- **GRPO**：只需要比较同一道题的多个答案，哪个更好

---

## 组内优势计算

### 传统优势估计

传统 PPO 用 Critic 网络计算优势：

$$A_t = Q(s_t, a_t) - V(s_t)$$

需要训练 Critic 网络来估计 $V(s_t)$。

### GRPO 优势估计

GRPO 用**组内相对优势**：

对每个 prompt $x$，采样 $G$ 个 response $\{y_1, y_2, ..., y_G\}$，计算奖励 $\{r_1, r_2, ..., r_G\}$。

优势函数：

$$A_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$

### 代码实现

```python
import numpy as np

def compute_group_advantages(rewards, group_size=4):
    """
    计算组内相对优势

    A_i = (r_i - mean(r)) / std(r)
    """
    rewards = np.array(rewards)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # 标准化
    advantages = (rewards - mean_reward) / (std_reward + 1e-8)

    return advantages
```

### 示例

```python
rewards = [0.8, 0.6, 0.9, 0.5]
advantages = compute_group_advantages(rewards)
# [0.45, -0.45, 1.35, -1.35]
```

解释：
- 奖励 0.9 的 response：优势 1.35（最好）
- 奖励 0.5 的 response：优势 -1.35（最差）
- 奖励 0.8 的 response：优势 0.45（较好）
- 奖励 0.6 的 response：优势 -0.45（较差）

---

## GRPO 损失函数

### 公式

GRPO 的损失函数与 PPO 类似，但有两个关键区别：

1. **不需要 Critic 损失**：因为没有 Critic 网络
2. **添加 KL 惩罚**：防止策略偏离参考策略太远

$$L(\theta) = E\left[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t) - \beta \cdot KL(\pi_\theta || \pi_{ref})\right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(y|x)}{\pi_{\theta_{old}}(y|x)}$
- $A_t$：组内相对优势
- $\beta$：KL 惩罚系数
- $KL(\pi_\theta || \pi_{ref})$：当前策略与参考策略的 KL 散度

### KL 散度计算

对于 LLM，KL 散度可以简化为：

$$KL(\pi_\theta || \pi_{ref}) = E_{y \sim \pi_\theta}\left[\log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}\right]$$

在实践中，我们用采样的 response 来近似：

$$KL \approx \frac{1}{G} \sum_{i=1}^{G} \log \frac{\pi_\theta(y_i|x)}{\pi_{ref}(y_i|x)}$$

---

## 代码实现

### GRPO Trainer 类

```python
import torch
import torch.nn as nn
import numpy as np

class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, reward_model,
                 lr=1e-6, clip_epsilon=0.2, beta=0.1, group_size=4):
        """
        GRPO 训练器

        参数:
            model: 当前策略 π_θ
            ref_model: 参考策略 π_ref
            tokenizer: 分词器
            reward_model: 奖励模型
            lr: 学习率
            clip_epsilon: 裁剪参数
            beta: KL 惩罚系数
            group_size: 每个 prompt 采样的 response 数量
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.group_size = group_size

    def generate_responses(self, prompt, max_length=512):
        """
        对每个 prompt 采样多个 response

        返回: list of responses
        """
        responses = []
        for _ in range(self.group_size):
            response = self.model.generate(
                prompt, max_length=max_length,
                do_sample=True, temperature=0.7
            )
            responses.append(response)
        return responses

    def compute_rewards(self, prompt, responses):
        """
        计算每个 response 的奖励

        返回: list of rewards
        """
        return [self.reward_model(prompt, r) for r in responses]

    def compute_logprobs(self, model, prompt, response):
        """
        计算 log π(y|x)

        返回: scalar
        """
        inputs = self.tokenizer(prompt + response, return_tensors="pt")
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)

        return token_log_probs.mean()

    def grpo_loss(self, prompt, responses, advantages):
        """
        GRPO 损失函数

        L(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t) - β * KL(π_θ || π_ref)]
        """
        losses = []

        for i, response in enumerate(responses):
            # 当前策略的 log 概率
            new_logprobs = self.compute_logprobs(self.model, prompt, response)

            # 参考策略的 log 概率
            with torch.no_grad():
                old_logprobs = self.compute_logprobs(self.ref_model, prompt, response)

            # 计算比率
            ratio = torch.exp(new_logprobs - old_logprobs)

            # 裁剪目标
            surr1 = ratio * advantages[i]
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                1 + self.clip_epsilon) * advantages[i]
            actor_loss = -torch.min(surr1, surr2)

            # KL 惩罚
            kl_penalty = self.beta * (new_logprobs - old_logprobs)

            losses.append(actor_loss + kl_penalty)

        return torch.stack(losses).mean()

    def train_step(self, prompt):
        """
        一步训练

        1. 采样多个 response
        2. 计算奖励
        3. 计算组内优势
        4. 计算损失并更新
        """
        # 1. 采样多个 response
        responses = self.generate_responses(prompt)

        # 2. 计算奖励
        rewards = self.compute_rewards(prompt, responses)

        # 3. 计算组内优势
        advantages = compute_group_advantages(rewards)

        # 4. 计算损失
        loss = self.grpo_loss(prompt, responses, advantages)

        # 5. 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), np.mean(rewards)
```

### 训练循环

```python
def train_grpo(trainer, prompts, num_epochs=10):
    """
    GRPO 训练循环

    对每个 prompt 进行多轮训练
    """
    for epoch in range(num_epochs):
        total_loss = 0
        total_reward = 0

        for prompt in prompts:
            loss, reward = trainer.train_step(prompt)
            total_loss += loss
            total_reward += reward

        avg_loss = total_loss / len(prompts)
        avg_reward = total_reward / len(prompts)

        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}, Reward: {avg_reward:.4f}")
```

---

## 算法流程

```
1. 初始化策略 π_θ 和参考策略 π_ref
2. 对于每个训练步骤:
   a. 采样 prompt x
   b. 使用 π_θ 采样 G 个 response: {y_1, ..., y_G}
   c. 计算每个 response 的奖励: {r_1, ..., r_G}
   d. 计算组内优势: A_i = (r_i - mean(r)) / std(r)
   e. 计算 KL 散度惩罚: KL(π_θ || π_ref)
   f. 计算 GRPO 损失
   g. 更新策略 π_θ
```

---

## GRPO vs PPO

### 优势

| 特性 | PPO | GRPO |
|------|-----|------|
| Critic 网络 | 需要 | 不需要 |
| 显存占用 | 高 | 低 |
| 训练复杂度 | 高 | 低 |
| 适用场景 | 通用 | LLM 微调 |

### 劣势

| 特性 | PPO | GRPO |
|------|-----|------|
| 优势估计精度 | 高 | 中 |
| 方差 | 低 | 较高 |
| 理论保证 | 更强 | 较弱 |

---

## 深入讨论

### 为什么 GRPO 不需要 Critic？

传统 PPO 需要 Critic 来估计 $V(s)$，然后计算 $A_t = Q(s_t, a_t) - V(s_t)$。

GRPO 的关键洞察是：

> 对于 LLM 生成任务，同一个 prompt 的多个 response 之间进行相对比较，可以替代 Critic 的作用。

### 组内优势的方差

组内优势的方差取决于：
1. **组内奖励的方差**：方差越大，优势估计越不稳定
2. **组大小 $G$**：组越大，优势估计越稳定

### KL 惩罚的作用

KL 惩罚 $\beta \cdot KL(\pi_\theta || \pi_{ref})$ 的作用：

1. **防止策略偏离太远**：避免灾难性遗忘
2. **稳定训练**：避免策略崩溃
3. **保持多样性**：避免模式坍塌

### $\beta$ 的选择

- $\beta$ 太小：策略可能偏离参考策略太远
- $\beta$ 太大：策略更新太保守
- 推荐值：0.01-0.1

---

## 关键要点

1. **GRPO 不需要 Critic 网络**，减少显存占用
2. **组内相对比较**替代 Critic 的作用
3. **KL 惩罚**防止策略偏离太远
4. **专为 LLM 设计**，适合生成任务
5. **DeepSeek** 等模型使用 GRPO 进行训练

---

## 延伸阅读

1. **DeepSeek-R1**：使用 GRPO 训练的推理模型
2. **Reinforcement Learning from Human Feedback (RLHF)**：GRPO 的背景
3. **Online DPO**：结合在线采样和 DPO
