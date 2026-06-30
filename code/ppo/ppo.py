"""
近端策略优化 (PPO) 算法实现

PPO 是最流行的策略梯度算法之一，通过裁剪机制确保更新稳定性。

算法特点:
    1. 裁剪目标函数：限制策略更新幅度
    2. 广义优势估计 (GAE)：平衡偏差和方差
    3. 多轮更新：提高样本效率

安装依赖:
    pip install gym==0.25.2
    pip install torch
    pip install numpy
    pip install matplotlib
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gym
import numpy as np
import matplotlib.pyplot as plt


class ActorCritic(nn.Module):
    """Actor-Critic 网络"""

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


class PPO:
    """PPO 算法"""

    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, lam=0.95,
                 clip_epsilon=0.2, n_epochs=10, batch_size=64, c1=0.5, c2=0.01):
        """
        初始化 PPO

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            lam: GAE参数
            clip_epsilon: 裁剪范围
            n_epochs: 每次采集后的训练轮数
            batch_size: 小批量大小
            c1: 价值损失系数
            c2: 熵损失系数
        """
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.c1 = c1
        self.c2 = c2

    def compute_gae(self, rewards, values, next_values, dones):
        """
        计算广义优势估计 (GAE)

        A_t = ∑_{l=0}^∞ (γλ)^l δ_{t+l}
        其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)

        Args:
            rewards: 奖励序列
            values: 状态价值
            next_values: 下一状态价值
            dones: 是否结束标志

        Returns:
            advantages: 优势估计
            returns: 回报
        """
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = next_values[t]

            # TD误差
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            # GAE
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, states, actions, old_log_probs, advantages, returns):
        """
        PPO 更新

        Args:
            states: 状态序列
            actions: 动作序列
            old_log_probs: 旧的log概率
            advantages: 优势估计
            returns: 回报

        Returns:
            loss: 平均损失
        """
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


def collect_trajectories(env, policy, max_steps=2048):
    """
    收集轨迹数据

    Args:
        env: 环境
        policy: 策略网络
        max_steps: 最大步数

    Returns:
        trajectory: 轨迹数据字典
    """
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


def train_ppo(env_name='CartPole-v0', num_iterations=500, max_steps=2048):
    """
    PPO 训练主循环

    Args:
        env_name: 环境名称
        num_iterations: 训练迭代次数
        max_steps: 每次采集的最大步数

    Returns:
        episode_rewards: 奖励记录
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo = PPO(state_dim, action_dim)

    episode_rewards = []
    best_reward = 0

    print(f"环境: {env_name}")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    print("=" * 50)

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
        avg_reward = trajectory['total_reward'] / (max_steps / 200)
        episode_rewards.append(avg_reward)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")

        # 5. 保存最佳模型
        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(ppo.policy.state_dict(), 'best_ppo_model.pth')

    return episode_rewards


def test_model(model_path, num_episodes=10):
    """
    测试训练好的模型

    Args:
        model_path: 模型路径
        num_episodes: 测试episode数量
    """
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


def plot_training(rewards, window=100):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 5))

    # 原始奖励
    plt.plot(rewards, alpha=0.3, label='Raw')

    # 滑动平均
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg,
                label=f'Moving Avg ({window})')

    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('PPO Training')
    plt.legend()
    plt.grid(True)
    plt.savefig('ppo_training.png')
    plt.show()


def main():
    """主函数"""

    # 训练
    print("开始训练 PPO...")
    rewards = train_ppo(
        env_name='CartPole-v0',
        num_iterations=300,
        max_steps=2048
    )

    # 可视化
    plot_training(rewards)

    # 测试
    print("\n" + "=" * 50)
    print("测试最佳模型")
    print("=" * 50)
    test_model('best_ppo_model.pth')


if __name__ == "__main__":
    main()
