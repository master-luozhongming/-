# Actor-Critic 架构

## 核心思想

Actor-Critic 结合了策略梯度（Actor）和价值函数（Critic）：

- **Actor（演员）**：策略网络 $\pi_\theta(a|s)$，决定采取什么动作
- **Critic（评论家）**：价值网络 $V_\phi(s)$，评估状态的好坏

---

## 架构图

```
┌─────────────────────────────────────────────────────────┐
│                      Actor-Critic                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   状态 s ─────┬─────► Actor π_θ ─────► 动作 a          │
│               │                                         │
│               └─────► Critic V_φ ────► 价值 V(s)       │
│                                                         │
│   优势 A = R + γV(s') - V(s)                           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 与 REINFORCE 的区别

| 特性 | REINFORCE | Actor-Critic |
|------|-----------|--------------|
| 回报估计 | 蒙特卡洛 $G_t$ | TD 误差 $\delta_t$ |
| 偏差 | 无偏 | 有偏 |
| 方差 | 高 | 低 |
| 更新方式 | 回合结束才能更新 | 每步都能更新 |

---

## TD 误差

### 定义

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

### 与回报的关系

$$G_t \approx V_\phi(s_t) + \delta_t$$

更准确地说，TD 误差是回报的无偏估计：

$$E[\delta_t] = E[G_t - V_\phi(s_t)]$$

---

## 代码实现

### Actor-Critic 网络

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

    def get_action(self, state):
        action_probs, value = self.forward(torch.FloatTensor(state))
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
```

### 训练循环

```python
def actor_critic_train(env, model, optimizer, gamma=0.99, num_episodes=1000):
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # 选择动作
            action, log_prob, value = model.get_action(state)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 计算下一状态的价值
            with torch.no_grad():
                _, next_value = model(torch.FloatTensor(next_state))
                next_value = next_value if not done else torch.tensor(0.0)

            # 计算 TD 误差
            td_target = reward + gamma * next_value
            td_error = td_target - value

            # Actor 损失（策略梯度）
            actor_loss = -log_prob * td_error.detach()

            # Critic 损失（价值函数）
            critic_loss = td_error.pow(2)

            # 总损失
            loss = actor_loss + 0.5 * critic_loss

            # 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state
            total_reward += reward

        episode_rewards.append(total_reward)

        if episode % 100 == 0:
            print(f"Episode {episode}, Avg Reward: {np.mean(episode_rewards[-100:]):.2f}")

    return episode_rewards
```

---

## 蒙特卡洛 vs TD 方法

### 蒙特卡洛方法（REINFORCE）

- 使用完整轨迹的回报 $G_t$
- **无偏**，但**方差大**
- 需要回合结束才能更新

### TD 方法（Actor-Critic）

- 使用 TD 误差 $\delta_t = r + \gamma V(s') - V(s)$
- **有偏**，但**方差小**
- 每步都能更新

### 对比图

```
蒙特卡洛:  s₀ → a₀ → r₀ → s₁ → a₁ → r₁ → ... → s_T
                        └──────────── G₀ ────────────┘

TD 方法:   s₀ → a₀ → r₀ → s₁
                        └─ δ₀ = r₀ + γV(s₁) - V(s₀)
```

---

## 关键要点

1. **Actor-Critic** 结合策略梯度和价值函数
2. **TD 误差**减少方差，但引入偏差
3. **每步更新**提高样本效率
4. 是 **PPO** 等现代算法的基础
