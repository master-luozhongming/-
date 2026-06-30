# 原始策略梯度法

## 核心思想

策略梯度法直接参数化策略函数，通过梯度上升来优化累积奖励。

### 策略函数

$$\pi_\theta(a|s) = P(a_t = a | s_t = s; \theta)$$

其中 $\theta$ 是策略网络的参数。

### 目标函数

$$J(\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

我们的目标是最大化这个目标函数：$\max_\theta J(\theta)$

---

## 策略梯度定理

### 核心公式

$$\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

其中 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是从时刻 $t$ 开始的累积回报。

### 直观理解

- 如果某个动作获得了**高回报** ($G_t$ 大) → **增加**该动作的概率
- 如果某个动作获得了**低回报** ($G_t$ 小) → **减少**该动作的概率

---

## 推导过程

### 从期望到采样

$$J(\theta) = E_{\tau \sim \pi_\theta}[R(\tau)] = \int p(\tau; \theta) R(\tau) d\tau$$

其中 $p(\tau; \theta)$ 是轨迹 $\tau$ 的概率。

### 对数导数技巧

$$\nabla_\theta p(\tau; \theta) = p(\tau; \theta) \nabla_\theta \log p(\tau; \theta)$$

### 轨迹概率

$$p(\tau; \theta) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|s_t, a_t)$$

取对数：

$$\log p(\tau; \theta) = \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T} \log p(s_{t+1}|s_t, a_t)$$

### 求梯度

$$\nabla_\theta \log p(\tau; \theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

注意：$p(s_0)$ 和 $p(s_{t+1}|s_t, a_t)$ 与 $\theta$ 无关，梯度为 0。

### 最终公式

$$\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

---

## 代码实现

### 策略网络

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

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
```

### 策略梯度更新

```python
def policy_gradient_update(policy, optimizer, states, actions, returns):
    """
    策略梯度更新

    ∇J(θ) = ∑_t ∇log π_θ(a_t|s_t) · G_t
    """
    # 计算 log π_θ(a_t|s_t)
    log_probs = []
    for state, action in zip(states, actions):
        probs = policy(torch.FloatTensor(state))
        dist = Categorical(probs)
        log_probs.append(dist.log_prob(torch.tensor(action)))

    # 计算损失：-∑_t log π_θ(a_t|s_t) · G_t
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss -= log_prob * G

    # 梯度上升 → 最小化负损失
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

---

## 深入讨论

### 为什么是梯度上升？

我们希望**最大化**目标函数 $J(\theta)$，所以使用梯度**上升**：

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

在代码中，我们通常最小化 $-J(\theta)$，等价于梯度上升。

### 方差问题

原始策略梯度法的主要问题是**方差大**：

1. 蒙特卡洛估计本身方差大
2. 不同轨迹的回报差异大
3. 导致收敛慢且不稳定

### 解决方案

1. **基线 (Baseline)**: 减去一个基线值
2. **优势函数**: 使用 $A(s,a) = Q(s,a) - V(s)$
3. **标准化回报**: 对回报进行标准化

---

## 关键要点

1. **策略梯度法直接优化策略**，不需要价值函数
2. **策略梯度定理**是所有策略梯度算法的基础
3. **方差大**是主要问题，需要使用基线等技巧
4. **REINFORCE** 是最简单的策略梯度算法实现

---

<style>
.math-block {
  text-align: center;
  margin: 1em 0;
}
</style>
