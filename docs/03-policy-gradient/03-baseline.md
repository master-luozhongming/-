# 带基线的策略梯度法

## 为什么需要基线？

### 方差问题

原始策略梯度的梯度估计：

$$\nabla_\theta J(\theta) = E \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

问题：$G_t$ 的方差很大，导致梯度估计不稳定。

### 基线的作用

引入基线 $b(s_t)$：

$$\nabla_\theta J(\theta) = E \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot (G_t - b(s_t)) \right]$$

**关键性质**：减去基线不会改变梯度的期望，但可以减少方差。

---

## 数学证明

### 为什么基线不影响期望？

$$E \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t) \right] = 0$$

证明：

$$E_{a_t \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t) \right]$$

$$= \sum_{a_t} \pi_\theta(a_t|s_t) \cdot \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_\theta(a_t|s_t)} \cdot b(s_t)$$

$$= \sum_{a_t} \nabla_\theta \pi_\theta(a_t|s_t) \cdot b(s_t)$$

$$= b(s_t) \cdot \nabla_\theta \sum_{a_t} \pi_\theta(a_t|s_t)$$

$$= b(s_t) \cdot \nabla_\theta 1 = 0$$

---

## 常用基线选择

### 1. 状态价值函数 $V(s)$

最常用的基线：

$$b(s_t) = V_{\phi}(s_t)$$

此时优势函数为：

$$A_t = G_t - V_{\phi}(s_t)$$

### 2. 移动平均回报

```python
class MovingAverageBaseline:
    def __init__(self, window_size=100):
        self.returns = []
        self.window_size = window_size

    def update(self, G):
        self.returns.append(G)
        if len(self.returns) > self.window_size:
            self.returns.pop(0)

    def get_baseline(self):
        return np.mean(self.returns)
```

### 3. 回报均值

```python
# 对当前batch的回报取均值作为基线
baseline = returns.mean()
advantages = returns - baseline
```

---

## 代码实现

### 带基线的 REINFORCE

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


class ValueNetwork(nn.Module):
    """基线网络（状态价值函数）"""
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return x.squeeze(-1)


def reinforce_with_baseline(env, policy, value_net, optimizer_policy,
                            optimizer_value, gamma=0.99, num_episodes=1000):
    """带基线的 REINFORCE"""
    episode_rewards = []

    for episode in range(num_episodes):
        states = []
        log_probs = []
        rewards = []

        state = env.reset()
        done = False

        while not done:
            action, log_prob = policy.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # 计算回报
        returns = compute_returns(rewards, gamma)

        # 计算基线（状态价值）
        states_tensor = torch.FloatTensor(np.array(states))
        values = value_net(states_tensor)

        # 计算优势：A_t = G_t - V(s_t)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 策略损失
        policy_loss = 0
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss -= log_prob * advantage

        # 价值网络损失
        value_loss = nn.MSELoss()(values, returns)

        # 更新策略
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()

        # 更新价值网络
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

        episode_rewards.append(sum(rewards))

        if episode % 100 == 0:
            print(f"Episode {episode}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}")

    return episode_rewards
```

---

## 基线的方差减少效果

### 理论分析

假设我们有 $N$ 个样本，梯度估计的方差为：

$$Var[\hat{g}] = \frac{1}{N} Var[\nabla_\theta \log \pi_\theta(a|s) \cdot (G - b)]$$

最优基线：

$$b^* = \frac{E[(\nabla_\theta \log \pi_\theta)^2 \cdot G]}{E[(\nabla_\theta \log \pi_\theta)^2]}$$

### 实验对比

```python
# 无基线
rewards_basic = reinforce(env, policy, optimizer)

# 有基线
rewards_baseline = reinforce_with_baseline(env, policy, value_net, ...)

# 通常有基线的版本收敛更快、更稳定
```

---

## 关键要点

1. **基线减少方差**，不改变梯度期望
2. **状态价值函数**是最常用的基线
3. **带基线的策略梯度**是 Actor-Critic 的基础
4. **优势函数** $A = G - V$ 衡量动作的好坏程度
