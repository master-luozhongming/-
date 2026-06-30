# REINFORCE 算法

## 算法原理

REINFORCE 是最基础的策略梯度算法，通过采样完整轨迹来估计策略梯度。

### 算法流程

```
1. 初始化策略参数 θ
2. 对于每个episode:
   a. 使用当前策略采样一条完整轨迹 τ
   b. 计算每个时刻的回报 G_t
   c. 计算策略梯度: ∇J(θ) = ∑_t ∇log π_θ(a_t|s_t) · G_t
   d. 更新参数: θ ← θ + α∇J(θ)
```

---

## 代码实现

### 完整实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

    def get_action(self, state):
        probs = self.forward(torch.FloatTensor(state))
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)


def compute_returns(rewards, gamma=0.99):
    """计算回报 G_t = R_t + γR_{t+1} + γ²R_{t+2} + ..."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.FloatTensor(returns)


def reinforce(env, policy, optimizer, gamma=0.99, num_episodes=1000):
    """REINFORCE 算法"""
    episode_rewards = []

    for episode in range(num_episodes):
        # 收集轨迹
        log_probs = []
        rewards = []

        state = env.reset()
        done = False

        while not done:
            action, log_prob = policy.get_action(state)
            next_state, reward, done, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # 计算回报
        returns = compute_returns(rewards, gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 计算损失
        loss = 0
        for log_prob, G in zip(log_probs, returns):
            loss -= log_prob * G

        # 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(sum(rewards))

        if episode % 100 == 0:
            print(f"Episode {episode}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}")

    return episode_rewards
```

### 训练脚本

```python
def main():
    env = gym.make("CartPole-v0")
    policy = PolicyNetwork(4, 2)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    rewards = reinforce(env, policy, optimizer)

if __name__ == "__main__":
    main()
```

---

## 优缺点

### 优点

- ✅ 简单直观，易于实现
- ✅ 无偏估计
- ✅ 适用于连续动作空间

### 缺点

- ❌ 方差大，收敛慢
- ❌ 样本效率低
- ❌ 只能用于回合制任务

---

## 变体

### 1. 带基线的 REINFORCE

```python
# 使用状态价值作为基线
advantages = returns - values.detach()
loss = -log_probs * advantages
```

### 2. 带熵正则化的 REINFORCE

```python
# 鼓励探索
entropy = -torch.sum(probs * torch.log(probs + 1e-8))
loss = policy_loss - beta * entropy
```

---

## 关键要点

1. REINFORCE 是最基础的策略梯度算法
2. 方差大是主要问题
3. 基线可以有效减少方差
4. 是理解 Actor-Critic 和 PPO 的基础
