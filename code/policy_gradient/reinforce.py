"""
REINFORCE 算法实现

REINFORCE 是最基础的策略梯度算法，通过采样完整轨迹来估计策略梯度。

算法流程:
    1. 初始化策略参数 θ
    2. 对于每个episode:
        a. 使用当前策略采样一条完整轨迹 τ
        b. 计算每个时刻的回报 G_t
        c. 计算策略梯度: ∇J(θ) = ∑_t ∇log π_θ(a_t|s_t) · G_t
        d. 更新参数: θ ← θ + α∇J(θ)

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


class PolicyNetwork(nn.Module):
    """策略网络"""

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


def compute_returns(rewards, gamma=0.99):
    """
    计算回报 G_t

    G_t = R_t + γR_{t+1} + γ²R_{t+2} + ...

    Args:
        rewards: 奖励序列
        gamma: 折扣因子

    Returns:
        returns: 回报序列
    """
    returns = []
    G = 0

    # 从后往前计算
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return torch.FloatTensor(returns)


def reinforce(env, policy, optimizer, gamma=0.99, num_episodes=1000):
    """
    REINFORCE 算法

    Args:
        env: 环境
        policy: 策略网络
        optimizer: 优化器
        gamma: 折扣因子
        num_episodes: 训练episode数量

    Returns:
        episode_rewards: 每个episode的总奖励
    """
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
        returns = compute_returns(rewards, gamma)

        # 标准化回报（减少方差）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # 计算策略梯度损失
        # ∇J(θ) = ∑_t ∇log π_θ(a_t|s_t) · G_t
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


def reinforce_with_baseline(env, policy, value_net, optimizer_policy,
                            optimizer_value, gamma=0.99, num_episodes=1000):
    """
    带基线的 REINFORCE 算法

    使用基线减少方差：
    ∇J(θ) = ∑_t ∇log π_θ(a_t|s_t) · (G_t - b(s_t))

    Args:
        env: 环境
        policy: 策略网络
        value_net: 价值网络（基线）
        optimizer_policy: 策略优化器
        optimizer_value: 价值网络优化器
        gamma: 折扣因子
        num_episodes: 训练episode数量

    Returns:
        episode_rewards: 每个episode的总奖励
    """
    episode_rewards = []

    for episode in range(num_episodes):
        # 收集轨迹
        states = []
        actions = []
        rewards = []
        log_probs = []

        state = env.reset()
        done = False

        while not done:
            action, log_prob = policy.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state

        # 计算回报
        returns = compute_returns(rewards, gamma)

        # 计算基线（状态价值）
        states_tensor = torch.FloatTensor(states)
        values = value_net(states_tensor).squeeze()

        # 计算优势函数
        advantages = returns - values.detach()

        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 策略损失（使用优势代替回报）
        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        policy_loss = torch.stack(policy_loss).sum()

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

        # 记录
        episode_reward = sum(rewards)
        episode_rewards.append(episode_reward)

        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

    return episode_rewards


def plot_training(rewards, title="REINFORCE Training", window=100):
    """绘制训练曲线"""
    plt.figure(figsize=(10, 5))

    # 原始奖励
    plt.plot(rewards, alpha=0.3, label='Raw')

    # 滑动平均
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), moving_avg,
                label=f'Moving Avg ({window})')

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.show()


def main():
    """主函数"""

    # 创建环境
    env = gym.make("CartPole-v0")
    state_dim = env.observation_space.shape[0]  # 4
    action_dim = env.action_space.n              # 2

    print(f"环境: CartPole-v0")
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")

    # ============ 基础 REINFORCE ============
    print("\n" + "=" * 50)
    print("训练基础 REINFORCE")
    print("=" * 50)

    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    rewards_basic = reinforce(
        env, policy, optimizer,
        gamma=0.99,
        num_episodes=500
    )

    plot_training(rewards_basic, "Basic REINFORCE")

    # ============ 带基线的 REINFORCE ============
    print("\n" + "=" * 50)
    print("训练带基线的 REINFORCE")
    print("=" * 50)

    policy_bl = PolicyNetwork(state_dim, action_dim)
    value_net = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    optimizer_policy = optim.Adam(policy_bl.parameters(), lr=0.001)
    optimizer_value = optim.Adam(value_net.parameters(), lr=0.001)

    rewards_baseline = reinforce_with_baseline(
        env, policy_bl, value_net,
        optimizer_policy, optimizer_value,
        gamma=0.99,
        num_episodes=500
    )

    plot_training(rewards_baseline, "REINFORCE with Baseline")

    # ============ 比较结果 ============
    print("\n" + "=" * 50)
    print("结果比较")
    print("=" * 50)

    print(f"\n基础 REINFORCE:")
    print(f"  最后100个episode平均奖励: {np.mean(rewards_basic[-100:]):.2f}")
    print(f"  最高奖励: {np.max(rewards_basic):.2f}")

    print(f"\n带基线的 REINFORCE:")
    print(f"  最后100个episode平均奖励: {np.mean(rewards_baseline[-100:]):.2f}")
    print(f"  最高奖励: {np.max(rewards_baseline):.2f}")


if __name__ == "__main__":
    main()
