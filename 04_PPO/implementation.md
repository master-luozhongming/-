# PPO 实战实现

## 网络结构定义

### Actor-Critic 网络

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import gym
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor (策略网络)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic (价值网络)
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
        """采样动作"""
        action_probs, value = self.forward(torch.FloatTensor(state))
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def evaluate(self, states, actions):
        """评估动作（用于更新）"""
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(), entropy
```

---

## PPO 算法实现

### PPO 更新类

```python
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_epsilon=0.2, n_epochs=10, batch_size=64, c1=0.5, c2=0.01):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.c1 = c1  # 价值损失系数
        self.c2 = c2  # 熵损失系数

    def compute_gae(self, rewards, values, next_values, dones):
        """计算广义优势估计"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = next_values[t]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, states, actions, old_log_probs, advantages, returns):
        """PPO更新"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)

        total_loss_epoch = 0

        for epoch in range(self.n_epochs):
            # 随机打乱数据
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            # 小批量更新
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # 评估当前策略
                new_log_probs, values, entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )

                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # 裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失
                critic_loss = nn.MSELoss()(values, batch_returns)

                # 熵损失
                entropy_loss = -entropy.mean()

                # 总损失
                loss = actor_loss + self.c1 * critic_loss + self.c2 * entropy_loss

                # 更新
                self.optimizer.zero_grad()
                loss.backward()
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss_epoch += loss.item()

        return total_loss_epoch / self.n_epochs
```

---

## 完整训练循环

### 收集轨迹数据

```python
def collect_trajectories(env, policy, max_steps=2048):
    """收集轨迹数据"""
    states = []
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []

    state = env.reset()
    done = False
    total_reward = 0

    for step in range(max_steps):
        # 采样动作
        action, log_prob, value = policy.get_action(state)

        # 执行动作
        next_state, reward, done, _ = env.step(action)

        # 存储数据
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.item())
        values.append(value.item())

        state = next_state
        total_reward += reward

        if done:
            state = env.reset()
            done = False

    # 获取下一个状态的价值（用于GAE计算）
    _, _, next_value = policy.get_action(state)
    next_values = values[1:] + [next_value.item()]

    return {
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'dones': np.array(dones),
        'log_probs': np.array(log_probs),
        'values': np.array(values),
        'next_values': np.array(next_values),
        'total_reward': total_reward
    }
```

### 训练主循环

```python
def train_ppo(env_name='CartPole-v0', num_iterations=1000, max_steps=2048):
    """PPO训练主循环"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo = PPO(state_dim, action_dim)

    episode_rewards = []
    best_reward = 0

    for iteration in range(num_iterations):
        # 1. 收集轨迹数据
        trajectory = collect_trajectories(env, ppo.policy, max_steps)

        # 2. 计算GAE
        advantages, returns = ppo.compute_gae(
            trajectory['rewards'],
            trajectory['values'],
            trajectory['next_values'],
            trajectory['dones']
        )

        # 3. PPO更新
        loss = ppo.update(
            trajectory['states'],
            trajectory['actions'],
            trajectory['log_probs'],
            advantages,
            returns
        )

        # 4. 记录和打印
        avg_reward = trajectory['total_reward'] / (max_steps / 200)  # 估计episode数
        episode_rewards.append(avg_reward)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")

        # 5. 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(ppo.policy.state_dict(), 'best_ppo_model.pth')

    return episode_rewards
```

---

## 广义优势估计的计算

### 详细实现

```python
def compute_gae_detailed(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    详细计算GAE

    Args:
        rewards: 奖励序列 [r_0, r_1, ..., r_T]
        values: 状态价值 [V(s_0), V(s_1), ..., V(s_T)]
        next_values: 下一状态价值 [V(s_1), V(s_2), ..., V(s_{T+1})]
        dones: 结束标志 [done_0, done_1, ..., done_T]
        gamma: 折扣因子
        lam: GAE参数

    Returns:
        advantages: 优势估计
        returns: 回报
    """
    T = len(rewards)
    advantages = np.zeros(T)
    gae = 0

    # 从后往前计算
    for t in reversed(range(T)):
        if t == T - 1:
            # 最后一步
            next_value = 0 if dones[t] else next_values[t]
        else:
            next_value = values[t + 1]

        # TD误差
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # GAE
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    # 计算回报
    returns = advantages + values

    # 标准化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return torch.FloatTensor(advantages), torch.FloatTensor(returns)
```

### 使用示例

```python
# 假设已经收集了轨迹数据
rewards = [1, 1, 1, 1, 1]  # 每步奖励
values = [0.5, 0.6, 0.7, 0.8, 0.9]  # 状态价值
next_values = [0.6, 0.7, 0.8, 0.9, 0.0]  # 下一状态价值
dones = [0, 0, 0, 0, 1]  # 是否结束

advantages, returns = compute_gae_detailed(rewards, values, next_values, dones)
print(f"Advantages: {advantages}")
print(f"Returns: {returns}")
```

---

## 超参数调优

### 常见问题及解决方案

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 收藏太慢 | 学习率太低 | 增加学习率 |
| 不稳定 | clip_epsilon太大 | 减小clip_epsilon |
| 过拟合 | n_epochs太多 | 减少n_epochs |
| 探索不足 | 熵系数太小 | 增加c2 |

### 推荐配置

```python
# CartPole 环境
config = {
    'lr': 3e-4,
    'gamma': 0.99,
    'lam': 0.95,
    'clip_epsilon': 0.2,
    'n_epochs': 10,
    'batch_size': 64,
    'c1': 0.5,
    'c2': 0.01,
    'max_steps': 2048
}

# Atari 游戏
config_atari = {
    'lr': 2.5e-4,
    'gamma': 0.99,
    'lam': 0.95,
    'clip_epsilon': 0.1,
    'n_epochs': 4,
    'batch_size': 256,
    'c1': 0.5,
    'c2': 0.01,
    'max_steps': 128
}
```

---

## 可视化训练过程

```python
import matplotlib.pyplot as plt

def plot_training(rewards, window=100):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 5))

    # 原始奖励
    plt.plot(rewards, alpha=0.3, label='Raw')

    # 滑动平均
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg, label=f'Moving Avg ({window})')

    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('PPO Training')
    plt.legend()
    plt.grid(True)
    plt.show()
```

---

## 完整训练脚本

```python
def main():
    # 训练
    rewards = train_ppo(
        env_name='CartPole-v0',
        num_iterations=500,
        max_steps=2048
    )

    # 可视化
    plot_training(rewards)

    # 测试
    test_model('best_ppo_model.pth')

def test_model(model_path, num_episodes=10):
    """测试训练好的模型"""
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = ActorCritic(state_dim, action_dim)
    policy.load_state_dict(torch.load(model_path))
    policy.eval()

    rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            with torch.no_grad():
                action, _, _ = policy.get_action(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward}")

    print(f"\nAverage Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

if __name__ == "__main__":
    main()
```

---

## 关键要点

1. **PPO 是最常用的策略梯度算法之一**
2. **裁剪机制确保更新稳定性**
3. **GAE 平衡偏差和方差**
4. **超参数需要根据具体任务调整**
5. **可以使用 TensorBoard 进行可视化**

---

*参考: 原文档 Chapter 7*
