# 组相对策略优化 (GRPO)

## GRPO 原理

### 核心思想

GRPO (Group Relative Policy Optimization) 是一种专门为大语言模型设计的强化学习算法，通过组内相对比较来估计优势函数。

### 与 PPO 的区别

| 特性 | PPO | GRPO |
|------|-----|------|
| 优势估计 | 使用价值网络 | 组内相对比较 |
| 额外网络 | 需要 Critic 网络 | 不需要 |
| 训练复杂度 | 较高 | 较低 |
| 适用场景 | 通用 | LLM微调 |

### 数学公式

```
L(θ) = E[ min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t ) - β·KL(π_θ || π_ref) ]
```

其中：
- r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
- A_t: 组内相对优势
- β: KL惩罚系数

---

## 组内优势计算

### 核心概念

对于每个 prompt，采样多个 response，然后计算组内相对优势：

```python
def compute_group_advantages(rewards, group_size=4):
    """
    计算组内相对优势

    Args:
        rewards: 每个response的奖励 [r_1, r_2, ..., r_G]
        group_size: 组大小

    Returns:
        advantages: 归一化的优势
    """
    rewards = np.array(rewards)

    # 计算组内均值和标准差
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # 归一化优势
    advantages = (rewards - mean_reward) / (std_reward + 1e-8)

    return advantages
```

### 示例

```python
# 采样4个response
group_size = 4
rewards = [0.8, 0.6, 0.9, 0.5]

# 计算组内优势
advantages = compute_group_advantages(rewards, group_size)
print(f"Advantages: {advantages}")
# 输出: [ 0.45, -0.45,  1.35, -1.35]
```

---

## GRPO 算法流程

### 完整流程

```
1. 初始化策略 π_θ 和参考策略 π_ref
2. 对于每个训练步骤:
   a. 从数据集中采样 prompt x
   b. 使用 π_θ 采样 G 个 response: {y_1, y_2, ..., y_G}
   c. 计算每个 response 的奖励: {r_1, r_2, ..., r_G}
   d. 计算组内优势: A_i = (r_i - mean(r)) / std(r)
   e. 计算 KL 散度: KL(π_θ || π_ref)
   f. 更新策略: maximize Σ_i [min(r_i*A_i, clip(r_i)*A_i) - β·KL]
```

---

## 代码实现

### GRPO 训练器

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class GRPOTrainer:
    def __init__(self, model, ref_model, tokenizer, reward_model,
                 lr=1e-6, gamma=0.99, lam=0.95, clip_epsilon=0.2,
                 beta=0.1, group_size=4):
        self.model = model  # 策略模型
        self.ref_model = ref_model  # 参考模型
        self.tokenizer = tokenizer
        self.reward_model = reward_model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.beta = beta
        self.group_size = group_size

    def generate_responses(self, prompt, max_length=512):
        """采样多个response"""
        responses = []
        for _ in range(self.group_size):
            response = self.model.generate(
                prompt,
                max_length=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            responses.append(response)
        return responses

    def compute_rewards(self, prompt, responses):
        """计算奖励"""
        rewards = []
        for response in responses:
            reward = self.reward_model(prompt, response)
            rewards.append(reward)
        return rewards

    def compute_group_advantages(self, rewards):
        """计算组内优势"""
        rewards = np.array(rewards)
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        advantages = (rewards - mean_reward) / (std_reward + 1e-8)
        return torch.FloatTensor(advantages)

    def compute_kl_penalty(self, prompt, responses):
        """计算KL散度惩罚"""
        kl_penalties = []
        for response in responses:
            # 计算当前策略的概率
            with torch.no_grad():
                old_logprobs = self.compute_logprobs(self.model, prompt, response)

            # 计算参考策略的概率
            ref_logprobs = self.compute_logprobs(self.ref_model, prompt, response)

            # KL散度
            kl = old_logprobs - ref_logprobs
            kl_penalties.append(kl)

        return torch.stack(kl_penalties)

    def compute_logprobs(self, model, prompt, response):
        """计算log概率"""
        inputs = self.tokenizer(prompt + response, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs["input_ids"][:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)

        return token_log_probs.mean()

    def grpo_loss(self, prompt, responses, advantages):
        """计算GRPO损失"""
        losses = []

        for i, response in enumerate(responses):
            # 计算当前策略的log概率
            new_logprobs = self.compute_logprobs(self.model, prompt, response)

            # 计算旧策略的log概率（使用参考模型）
            with torch.no_grad():
                old_logprobs = self.compute_logprobs(self.ref_model, prompt, response)

            # 计算比率
            ratio = torch.exp(new_logprobs - old_logprobs)

            # 裁剪目标
            surr1 = ratio * advantages[i]
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                1 + self.clip_epsilon) * advantages[i]
            actor_loss = -torch.min(surr1, surr2)

            # KL惩罚
            kl_penalty = self.beta * (new_logprobs - old_logprobs)

            losses.append(actor_loss + kl_penalty)

        return torch.stack(losses).mean()

    def train_step(self, prompt):
        """单步训练"""
        # 1. 采样多个response
        responses = self.generate_responses(prompt)

        # 2. 计算奖励
        rewards = self.compute_rewards(prompt, responses)

        # 3. 计算组内优势
        advantages = self.compute_group_advantages(rewards)

        # 4. 计算损失
        loss = self.grpo_loss(prompt, responses, advantages)

        # 5. 更新
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item(), np.mean(rewards)
```

---

## 使用 GRPO 玩倒立摆游戏

### 环境设置

```python
import gym

def create_env():
    env = gym.make("CartPole-v0")
    return env

def state_to_prompt(state):
    """将状态转换为文本prompt"""
    return f"State: position={state[0]:.2f}, velocity={state[1]:.2f}, angle={state[2]:.2f}, angular_velocity={state[3]:.2f}"

def action_to_response(action):
    """将动作转换为文本response"""
    return "left" if action == 0 else "right"
```

### GRPO 训练

```python
def train_grpo_cartpole():
    env = create_env()

    # 简化的策略网络
    class SimplePolicy(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            )

        def forward(self, state):
            return self.fc(torch.FloatTensor(state))

        def get_action(self, state):
            probs = self.forward(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return action.item(), dist.log_prob(action)

    policy = SimplePolicy(4, 2)
    ref_policy = SimplePolicy(4, 2)
    ref_policy.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=1e-3)

    # 训练循环
    for episode in range(1000):
        state = env.reset()
        states = []
        actions = []
        rewards = []

        # 采样轨迹
        done = False
        while not done:
            action, _ = policy.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

        # 计算组内优势（将轨迹分成多个组）
        group_size = 4
        total_reward = sum(rewards)

        # 简化：使用整个轨迹的奖励
        advantages = torch.FloatTensor([(total_reward - 100) / 50])  # 归一化

        # 计算损失
        loss = 0
        for i, (state, action) in enumerate(zip(states, actions)):
            probs = policy(state)
            ref_probs = ref_policy(state)

            ratio = probs[action] / (ref_probs[action] + 1e-8)
            surr1 = ratio * advantages[0]
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages[0]
            actor_loss = -torch.min(surr1, surr2)

            kl = torch.sum(probs * torch.log(probs / (ref_probs + 1e-8)))
            loss += actor_loss + 0.1 * kl

        loss /= len(states)

        # 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward}")

train_grpo_cartpole()
```

---

## GRPO 在 LLM 中的应用

### 奖励模型

```python
class RewardModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids, attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        reward = self.reward_head(hidden_states[:, -1, :])
        return reward
```

### 训练流程

```python
def train_grpo_llm():
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    reward_model = RewardModel("gpt2")

    trainer = GRPOTrainer(model, ref_model, tokenizer, reward_model)

    # 训练循环
    for step in range(1000):
        prompt = "The meaning of life is"
        loss, avg_reward = trainer.train_step(prompt)

        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}, Avg Reward: {avg_reward:.4f}")
```

---

## GRPO 的优势

### 相比 PPO
1. **不需要价值网络**: 减少计算开销
2. **更稳定**: 组内比较减少方差
3. **更适合 LLM**: 专门为生成任务设计

### 相比 DPO
1. **不需要偏好数据**: 只需要奖励信号
2. **更灵活**: 可以使用任何奖励模型
3. **在线学习**: 可以持续改进

---

## 超参数调优

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| group_size | 4-16 | 采样response数量 |
| beta | 0.01-0.1 | KL惩罚系数 |
| clip_epsilon | 0.1-0.2 | 裁剪范围 |
| lr | 1e-6 - 1e-5 | 学习率 |

---

## 关键要点

1. **GRPO 是专门为 LLM 设计的 RL 算法**
2. **通过组内相对比较估计优势**
3. **不需要额外的价值网络**
4. **广泛应用于 RLHF**

---

*参考: 原文档 Chapter 9*
