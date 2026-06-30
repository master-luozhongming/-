# 组相对策略优化 (GRPO)

## 概述

GRPO 是专门为大语言模型设计的强化学习算法，通过组内相对比较来估计优势函数。

---

## 与 PPO 的对比

| 特性 | PPO | GRPO |
|------|-----|------|
| 优势估计 | 使用价值网络 | 组内相对比较 |
| 额外网络 | 需要 Critic | 不需要 |
| 训练复杂度 | 较高 | 较低 |
| 适用场景 | 通用 | LLM 微调 |

---

## 组内优势计算

### 核心思想

对每个 prompt 采样多个 response，计算组内相对优势：

```python
def compute_group_advantages(rewards, group_size=4):
    rewards = np.array(rewards)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    advantages = (rewards - mean_reward) / (std_reward + 1e-8)
    return advantages
```

### 示例

```python
rewards = [0.8, 0.6, 0.9, 0.5]
advantages = compute_group_advantages(rewards)
# [0.45, -0.45, 1.35, -1.35]
```

---

## GRPO 损失函数

$$L(\theta) = E\left[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t) - \beta \cdot KL(\pi_\theta || \pi_{ref})\right]$$

---

## 代码实现

```python
class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, reward_model,
                 lr=1e-6, clip_epsilon=0.2, beta=0.1, group_size=4):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.group_size = group_size

    def generate_responses(self, prompt, max_length=512):
        responses = []
        for _ in range(self.group_size):
            response = self.model.generate(
                prompt, max_length=max_length,
                do_sample=True, temperature=0.7
            )
            responses.append(response)
        return responses

    def compute_rewards(self, prompt, responses):
        return [self.reward_model(prompt, r) for r in responses]

    def grpo_loss(self, prompt, responses, advantages):
        losses = []

        for i, response in enumerate(responses):
            new_logprobs = self.compute_logprobs(self.model, prompt, response)
            with torch.no_grad():
                old_logprobs = self.compute_logprobs(self.ref_model, prompt, response)

            ratio = torch.exp(new_logprobs - old_logprobs)

            surr1 = ratio * advantages[i]
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                1 + self.clip_epsilon) * advantages[i]

            actor_loss = -torch.min(surr1, surr2)
            kl_penalty = self.beta * (new_logprobs - old_logprobs)

            losses.append(actor_loss + kl_penalty)

        return torch.stack(losses).mean()

    def train_step(self, prompt):
        responses = self.generate_responses(prompt)
        rewards = self.compute_rewards(prompt, responses)
        advantages = compute_group_advantages(rewards)
        loss = self.grpo_loss(prompt, responses, advantages)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), np.mean(rewards)
```

---

## 算法流程

```
1. 初始化策略 π_θ 和参考策略 π_ref
2. 对于每个训练步骤:
   a. 采样 prompt x
   b. 使用 π_θ 采样 G 个 response
   c. 计算每个 response 的奖励
   d. 计算组内优势
   e. 计算 KL 散度惩罚
   f. 更新策略
```

---

## 关键要点

1. **不需要价值网络**：减少计算开销
2. **组内相对比较**：减少方差
3. **专为 LLM 设计**：适合生成任务
4. **广泛应用**：DeepSeek 等模型使用 GRPO
