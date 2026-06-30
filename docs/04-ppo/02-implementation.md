# PPO 实现

## 为什么需要完整实现？

理解 PPO 的理论很重要，但**完整的实现**同样关键。本章将带你实现一个完整的 PPO 算法，包括：

1. **Actor-Critic 网络**：策略网络和价值网络
2. **GAE 计算**：广义优势估计
3. **PPO 损失函数**：裁剪目标 + 价值损失 + 熵损失
4. **训练循环**：数据采集 → 优势计算 → 多轮更新

---

## Actor-Critic 网络

### 架构设计

PPO 使用 Actor-Critic 架构：

- **Actor（策略网络）**：输出动作概率分布
- **Critic（价值网络）**：估计状态价值 $V(s)$

### 为什么共享特征层？

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # 共享特征层
        # 为什么共享？策略和价值函数都依赖于对状态的理解
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor（策略网络）
        # 输出每个动作的概率
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # 确保输出是概率分布
        )

        # Critic（价值网络）
        # 输出状态价值 V(s)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出标量
        )

    def forward(self, state):
        """前向传播"""
        features = self.shared(state)  # 提取特征
        action_probs = self.actor(features)  # 动作概率
        value = self.critic(features)  # 状态价值
        return action_probs, value

    def get_action(self, state):
        """采样动作"""
        action_probs, value = self.forward(torch.FloatTensor(state))

        # 创建分类分布
        dist = Categorical(action_probs)

        # 采样动作
        action = dist.sample()

        # 返回动作、log 概率、价值
        return action.item(), dist.log_prob(action), value

    def evaluate(self, states, actions):
        """评估动作"""
        action_probs, values = self.forward(states)

        # 创建分类分布
        dist = Categorical(action_probs)

        # 计算 log 概率
        log_probs = dist.log_prob(actions)

        # 计算熵（用于探索）
        entropy = dist.entropy()

        return log_probs, values.squeeze(), entropy
```

### 直观理解

```
状态 s
    ↓
┌───────────────┐
│ 共享特征层    │  提取状态特征
└───────────────┘
    ↓           ↓
┌───────────┐ ┌───────────┐
│ Actor     │ │ Critic    │
│ 策略网络  │ │ 价值网络  │
└───────────┘ └───────────┘
    ↓           ↓
π(a|s)      V(s)
```

---

## GAE 计算

### 为什么需要 GAE？

GAE（Generalized Advantage Estimation）用于计算优势函数 $A_t$：

$$A_t = Q(s_t, a_t) - V(s_t)$$

直接计算 $Q(s_t, a_t)$ 方差大，GAE 通过**指数加权平均**来平衡偏差和方差。

### GAE 公式

$$A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中：
- $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$（TD 误差）
- $\gamma$：折扣因子
- $\lambda$：GAE 参数，控制偏差-方差权衡

### 代码实现

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    计算广义优势估计 (GAE)

    A_t = ∑_{l=0}^{∞} (γλ)^l δ_{t+l}

    其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """
    advantages = []
    gae = 0

    # 从后往前计算
    for t in reversed(range(len(rewards))):
        # TD 误差
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]

        # GAE 递推公式
        gae = delta + gamma * lam * (1 - dones[t]) * gae

        # 插入到列表开头
        advantages.insert(0, gae)

    # 转换为 tensor
    advantages = torch.FloatTensor(advantages)

    # 计算回报（用于价值函数训练）
    returns = advantages + torch.FloatTensor(values)

    # 标准化优势（减少方差）
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns
```

### 直观理解

```
时间步 t:    s_t → a_t → r_t → s_{t+1}
TD 误差:     δ_t = r_t + γV(s_{t+1}) - V(s_t)

GAE:         A_t = δ_t + γλδ_{t+1} + (γλ)^2δ_{t+2} + ...

当 λ=0:     A_t = δ_t（只有当前 TD 误差，偏差大方差小）
当 λ=1:     A_t = G_t - V(s_t)（完整回报，偏差小方差大）
```

---

## PPO 损失函数

### 三重损失

PPO 的损失函数由三部分组成：

$$L(\theta) = L^{CLIP} - c_1 L^{VF} + c_2 S$$

其中：
- $L^{CLIP}$：裁剪的策略目标
- $L^{VF}$：价值函数损失
- $S$：熵正则化项

### 代码实现

```python
def ppo_loss(old_log_probs, new_log_probs, advantages, returns, values,
             clip_epsilon=0.2, c1=0.5, c2=0.01):
    """
    PPO 损失函数

    L(θ) = E[L^CLIP(θ) - c1 * L^VF(θ) + c2 * S[π_θ](s)]
    """
    # 计算比率 r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
    # 用 log 概率计算更稳定：r_t(θ) = exp(log π_θ(a|s) - log π_θ_old(a|s))
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 裁剪目标 L^CLIP
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    # 价值函数损失 L^VF
    critic_loss = nn.MSELoss()(values, returns)

    # 熵损失（鼓励探索）
    # 熵 = -∑ π(a|s) log π(a|s)
    entropy = -torch.sum(torch.exp(new_log_probs) * new_log_probs, dim=-1)
    entropy_loss = -entropy.mean()

    # 总损失
    total_loss = actor_loss + c1 * critic_loss + c2 * entropy_loss

    return total_loss, actor_loss, critic_loss, entropy_loss
```

### 为什么要用 log 概率？

直接计算 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 可能导致数值不稳定。

用 log 概率计算更稳定：

$$r_t(\theta) = \exp(\log \pi_\theta(a_t|s_t) - \log \pi_{\theta_{old}}(a_t|s_t))$$

---

## 完整的 PPO 算法

### 算法类

```python
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 lam=0.95, clip_epsilon=0.2, n_epochs=10, batch_size=64):
        """
        PPO 算法

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            lam: GAE 参数
            clip_epsilon: 裁剪参数
            n_epochs: 每批数据训练轮数
            batch_size: 批次大小
        """
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def compute_gae(self, rewards, values, next_values, dones):
        """计算 GAE"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, states, actions, old_log_probs, advantages, returns):
        """
        PPO 更新

        对同一批数据训练 n_epochs 轮
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)

        total_loss = 0

        for _ in range(self.n_epochs):
            # 打乱数据
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            # 分批训练
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                # 前向传播
                new_log_probs, values, entropy = self.policy.evaluate(
                    states[idx], actions[idx]
                )

                # 计算比率
                ratio = torch.exp(new_log_probs - old_log_probs[idx])

                # 裁剪目标
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * advantages[idx]

                # 计算损失
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, returns[idx])
                entropy_loss = -entropy.mean()

                loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss / self.n_epochs
```

### 训练循环

```python
def train_ppo(env, ppo, num_iterations=1000, max_steps=2048):
    """
    PPO 训练循环

    1. 采集轨迹
    2. 计算 GAE
    3. 多轮更新
    """
    episode_rewards = []

    for iteration in range(num_iterations):
        # 1. 采集轨迹
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        state = env.reset()
        for _ in range(max_steps):
            action, log_prob, value = ppo.policy.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state
            if done:
                state = env.reset()

        # 2. 计算 GAE
        advantages, returns = ppo.compute_gae(
            rewards, values, values[1:] + [0], dones
        )

        # 3. 多轮更新
        loss = ppo.update(states, actions, log_probs, advantages, returns)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss:.4f}")

    return episode_rewards
```

---

## 训练技巧

### 1. 梯度裁剪

```python
# 防止梯度爆炸
nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
```

### 2. 学习率调度

```python
# 随着训练进行，逐渐减小学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.95)
```

### 3. 优势标准化

```python
# 标准化优势，减少方差
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

### 4. 早停策略

```python
# 如果 KL 散度太大，提前停止更新
kl_div = (old_log_probs - new_log_probs).mean()
if kl_div > 0.02:
    break
```

---

## 关键要点

1. **Actor-Critic 架构**：共享特征层，减少计算开销
2. **GAE 计算**：平衡偏差和方差
3. **PPO 损失函数**：三重损失，平衡策略、价值、探索
4. **多轮更新**：提高样本效率
5. **训练技巧**：梯度裁剪、学习率调度、优势标准化

---

## 完整代码

完整的 PPO 实现可以在 GitHub 仓库的 `code/ppo.py` 文件中找到。
