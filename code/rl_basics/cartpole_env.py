"""
倒立摆环境基础操作示例

安装依赖:
    pip install gym==0.25.2
    pip install numpy
    pip install pygame
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

def basic_env_demo():
    """演示倒立摆环境的基本操作"""

    # 1. 创建环境
    print("=" * 50)
    print("1. 创建倒立摆环境")
    print("=" * 50)
    env = gym.make("CartPole-v0")

    # 2. 查看环境信息
    print(f"\nGym版本: {gym.__version__}")
    print(f"状态空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    print(f"状态维度: {env.observation_space.shape[0]}")
    print(f"动作数量: {env.action_space.n}")

    # 3. 重置环境
    print("\n" + "=" * 50)
    print("2. 重置环境获取初始状态")
    print("=" * 50)
    state = env.reset()
    print(f"初始状态 S0: {state}")
    print(f"  - 推车位置: {state[0]:.4f}")
    print(f"  - 推车速度: {state[1]:.4f}")
    print(f"  - 木杆角度: {state[2]:.4f}")
    print(f"  - 木杆角速度: {state[3]:.4f}")

    # 4. 执行动作
    print("\n" + "=" * 50)
    print("3. 执行动作")
    print("=" * 50)
    action = 0  # 0=向左推, 1=向右推
    print(f"执行动作: {'向左推' if action == 0 else '向右推'}")

    next_state, reward, done, info = env.step(action)
    print(f"\n执行结果:")
    print(f"  - 下一状态 S1: {next_state}")
    print(f"  - 奖励 R0: {reward}")
    print(f"  - 是否结束: {done}")
    print(f"  - 附加信息: {info}")

    return env


def random_agent_demo(num_episodes=5):
    """演示随机智能体"""

    print("\n" + "=" * 50)
    print("随机智能体演示")
    print("=" * 50)

    env = gym.make("CartPole-v0")
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0

        while not done:
            # 随机选择动作
            action = env.action_space.sample()

            # 执行动作
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            step += 1

            state = next_state

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: 总奖励 = {total_reward}, 步数 = {step}")

    print(f"\n平均奖励: {np.mean(episode_rewards):.2f}")
    print(f"标准差: {np.std(episode_rewards):.2f}")

    return episode_rewards


def run_multiple_steps(num_steps=20):
    """演示多步执行"""

    print("\n" + "=" * 50)
    print(f"执行 {num_steps} 步演示")
    print("=" * 50)

    env = gym.make("CartPole-v0")
    state = env.reset()

    print(f"{'Step':<6} {'Action':<10} {'Reward':<10} {'Done':<10} {'Position':<12} {'Angle':<12}")
    print("-" * 60)

    for step in range(num_steps):
        # 交替执行左右动作
        action = step % 2
        next_state, reward, done, _ = env.step(action)

        print(f"{step:<6} {'左' if action == 0 else '右':<10} {reward:<10} {done:<10} "
              f"{next_state[0]:<12.4f} {next_state[2]:<12.4f}")

        if done:
            print(f"\n游戏在第 {step} 步结束!")
            break

        state = next_state

    return


def visualize_trajectory():
    """可视化轨迹"""

    print("\n" + "=" * 50)
    print("轨迹可视化")
    print("=" * 50)

    env = gym.make("CartPole-v0")
    state = env.reset()

    positions = []
    angles = []
    rewards = []
    actions = []

    done = False
    while not done:
        # 使用简单策略：如果杆子向左偏，就向左推
        action = 0 if state[2] < 0 else 1

        next_state, reward, done, _ = env.step(action)

        positions.append(state[0])
        angles.append(state[2])
        rewards.append(reward)
        actions.append(action)

        state = next_state

    # 绘制图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 推车位置
    axes[0, 0].plot(positions)
    axes[0, 0].set_title('Cart Position')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].grid(True)

    # 木杆角度
    axes[0, 1].plot(angles)
    axes[0, 1].set_title('Pole Angle')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Angle (rad)')
    axes[0, 1].grid(True)

    # 奖励
    axes[1, 0].plot(rewards)
    axes[1, 0].set_title('Rewards')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Reward')
    axes[1, 0].grid(True)

    # 动作
    axes[1, 1].plot(actions)
    axes[1, 1].set_title('Actions')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Action (0=Left, 1=Right)')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('trajectory.png')
    plt.show()

    print(f"轨迹长度: {len(positions)} 步")
    print(f"总奖励: {sum(rewards)}")


def policy_comparison():
    """比较不同策略"""

    print("\n" + "=" * 50)
    print("策略比较")
    print("=" * 50)

    env = gym.make("CartPole-v0")

    # 策略1: 随机策略
    def random_policy(state):
        return env.action_space.sample()

    # 策略2: 基于角度的策略
    def angle_policy(state):
        return 0 if state[2] < 0 else 1

    # 策略3: 基于角速度的策略
    def angular_velocity_policy(state):
        if state[2] < -0.1:
            return 0
        elif state[2] > 0.1:
            return 1
        else:
            return 0 if state[3] < 0 else 1

    policies = {
        'Random': random_policy,
        'Angle-based': angle_policy,
        'Angular Velocity': angular_velocity_policy
    }

    results = {}
    num_episodes = 50

    for name, policy in policies.items():
        episode_rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = policy(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward

            episode_rewards.append(total_reward)

        results[name] = {
            'mean': np.mean(episode_rewards),
            'std': np.std(episode_rewards),
            'max': np.max(episode_rewards),
            'min': np.min(episode_rewards)
        }

    # 打印结果
    print(f"\n{'策略':<20} {'平均奖励':<12} {'标准差':<12} {'最大值':<12} {'最小值':<12}")
    print("-" * 68)
    for name, stats in results.items():
        print(f"{name:<20} {stats['mean']:<12.2f} {stats['std']:<12.2f} "
              f"{stats['max']:<12.2f} {stats['min']:<12.2f}")

    return results


if __name__ == "__main__":
    # 运行所有演示
    env = basic_env_demo()
    random_agent_demo()
    run_multiple_steps()
    visualize_trajectory()
    policy_comparison()
