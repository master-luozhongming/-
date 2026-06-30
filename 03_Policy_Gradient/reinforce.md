# REINFORCE 算法

## 算法原理

### 核心思想
REINFORCE 是一种蒙特卡洛策略梯度算法，通过采样完整轨迹来估计策略梯度。

### 策略梯度定理
```
∇J(θ) = E_τ[∑_{t=0}^T ∇log π_θ(a_t|s_t) · G_t]
```

其中：
- π_θ(a_t|s_t): 策略函数（神经网络）
- G_t: 从t时刻起的累积回报
- ∇log π_θ(a_t|s_t): 策略的对数梯度

### 直观理解
```
如果某个动作获得了高回报 → 增加该动作的概率
如果某个动作获得了低回报 → 减少该动作的概率
```

---

## 算法流程

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

### 策略网络

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

    def get_action(self, state):
        """采样动作并返回log概率"""
        probs = self.forward(torch.FloatTensor(state))
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob
```

### REINFORCE 算法

```python
def reinforce(env, policy, optimizer, gamma=0.99, num_episodes=1000):
    """REINFORCE算法实现"""

    episode_rewards = []

    for episode in range(num_episodes):
        # 收集一条完整轨迹
        states = []
        actions = []
        rewards = []
        log_probs = []

        state = env.reset()
        done = False

        while not done:
            # 采样动作
            action, log_prob = policy.get_action(state)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 存储数据
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state

        # 计算回报 G_t
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)

        # 标准化回报（减少方差）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 计算策略梯度损失
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()

        # 更新策略
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # 记录奖励
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

    return episode_rewards
```

### 完整训练代码

```python
import numpy as np

def main():
    # 创建环境
    env = gym.make("CartPole-v0")
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n              # 2

    # 创建策略网络
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    # 训练
    rewards = reinforce(env, policy, optimizer, gamma=0.99, num_episodes=1000)

    # 可视化
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('REINFORCE on CartPole')
    plt.show()

if __name__ == "__main__":
    main()
```

---

## REINFORCE 的变体

### 1. 带基线的 REINFORCE

使用基线减少方差：

```python
def reinforce_with_baseline(env, policy, value_net, optimizer_policy, optimizer_value, gamma=0.99):
    # ... 收集轨迹 ...

    # 计算回报
    returns = compute_returns(rewards, gamma)

    # 计算优势函数
    values = value_net(torch.FloatTensor(states))
    advantages = returns - values.detach()

    # 策略损失（使用优势代替回报）
    policy_loss = -log_probs * advantages

    # 价值网络损失
    value_loss = nn.MSELoss()(values, returns)

    # 更新
    optimizer_policy.zero_grad()
    policy_loss.backward()
    optimizer_policy.step()

    optimizer_value.zero_grad()
    value_loss.backward()
    optimizer_value.step()
```

### 2. 带熵正则化的 REINFORCE

鼓励探索：

```python
def policy_loss_with_entropy(log_probs, returns, probs, beta=0.01):
    # 策略梯度损失
    pg_loss = -log_probs * returns

    # 熵正则化
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    entropy_loss = -beta * entropy

    return pg_loss + entropy_loss
```

---

## 优缺点分析

### 优点
1. **简单直观**: 易于理解和实现
2. **无偏估计**: 策略梯度的无偏估计
3. **适用于连续动作空间**: 可以处理连续动作

### 缺点
1. **方差大**: 蒙特卡洛估计方差大
2. **收敛慢**: 需要大量样本
3. **样本效率低**: 每条轨迹只用一次
4. **只能用于回合制任务**: 需要完整轨迹

---

## 改进方向

### 1. 减少方差
- 使用基线 (baseline)
- 使用优势函数 (advantage)
- 标准化回报

### 2. 提高样本效率
- 经验回放 (Experience Replay)
- 重要性采样 (Importance Sampling)

### 3. 结合价值函数
- Actor-Critic 架构
- GAE (广义优势估计)

---

## 与其他算法对比

| 算法 | 类型 | 方差 | 偏差 | 样本效率 |
|------|------|------|------|----------|
| REINFORCE | 蒙特卡洛 | 高 | 无 | 低 |
| Actor-Critic | TD学习 | 低 | 有 | 高 |
| PPO | 策略梯度 | 低 | 有 | 高 |

---

## 关键要点

1. **REINFORCE 是最基础的策略梯度算法**
2. **通过采样完整轨迹来估计梯度**
3. **方差大是主要问题，需要使用基线等技巧**
4. **是理解更复杂算法（如PPO）的基础**

---

## 练习

1. 实现带基线的 REINFORCE
2. 在 MountainCar 环境上测试 REINFORCE
3. 比较不同学习率对收敛的影响
4. 添加熵正则化并观察效果

---

*参考: 原文档 Chapter 5.2*
