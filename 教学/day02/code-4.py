import gymnasium as gym  # 导入 Gymnasium 库
import random  # 导入随机数库（未使用）
import matplotlib.pyplot as plt  # 导入绑图库
import matplotlib.animation as animation  # 导入动画库
from matplotlib import rc  # 导入配置
import numpy as np  # 导入 NumPy（未使用）
import torch  # 导入 PyTorch
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式 API
import torch.optim as optim  # 导入优化器
from torch.distributions import Categorical  # 导入分类分布


def show_animation(imgs):  # 动画显示函数
    """将图片序列转为动画并保存"""
    rc("animation", html="jshtml")  # 设置后端
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))  # 创建图形
    frames = []  # 帧列表

    text = ax.text(10, 20, "", fontsize=12, color="black")  # 文本（未使用）

    for i, img in enumerate(imgs):  # 遍历图片
        frame = [ax.imshow(img, animated=True)]  # 添加帧
        frame.append(ax.text(10, 20, f"Step: {i+1}", animated=True))  # 步数标注
        frames.append(frame)

    ax.axis("off")  # 关闭坐标轴

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)  # 创建动画

    ani.save("cartpole.mp4", writer="ffmpeg")  # 保存 MP4
    ani.save("cartpole.gif", writer="pillow")  # 保存 GIF

    plt.close(fig)  # 关闭图形
    return ani


def plot_loss(episode_list, return_list, filename):  # 绘制曲线
    """绘制奖励图像"""
    f = plt.figure()  # 创建图形
    plt.plot(episode_list, return_list)  # 绘制
    plt.xlabel("Episodes")  # x 标签
    plt.ylabel("Returns")  # y 标签
    plt.title("CartPole-v0")  # 标题
    plt.show()  # 显示
    f.savefig(filename, bbox_inches="tight")  # 保存


class PolicyNet(nn.Module):  # 策略网络
    """策略神经网络的结构"""

    def __init__(self, action_size):  # 构造函数
        super().__init__()
        self.l1 = nn.Linear(4, 128)  # 第一层
        self.l2 = nn.Linear(128, action_size)  # 第二层

    def forward(self, x):  # 前向传播
        x = F.relu(self.l1(x))  # ReLU
        x = F.softmax(self.l2(x), dim=1)  # Softmax
        return x


class Agent:  # 智能体
    def __init__(self):  # 构造函数
        self.gamma = 0.98  # 折扣因子
        self.lr = 0.0002  # 学习率
        self.action_size = 2  # 动作数
        self.pi = PolicyNet(self.action_size)  # 策略网络
        self.optimizer = optim.Adam(  # 优化器
            self.pi.parameters(),
            lr=self.lr
        )

    def get_action(self, state):  # 选择动作
        """输入参数：环境的状态"""
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)  # 概率分布
        m = Categorical(probs)  # 分类分布
        action = m.sample().item()  # 采样
        return action, probs

    def collect_trajectory(self, env):  # 收集轨迹
        """在环境 env 中采样一条轨迹"""
        state, _ = env.reset()  # 重置（Gymnasium API）
        states, actions, rewards = [], [], []  # 轨迹列表
        done = False  # 结束标志

        while not done:  # 循环
            action, _ = self.get_action(state)  # 选择动作
            next_state, reward, terminated, truncated, _ = env.step(action)  # 执行
            done = terminated or truncated  # 判断结束

            states.append(state)  # 保存状态
            actions.append(action)  # 保存动作
            rewards.append(reward)  # 保存奖励

            state = next_state  # 状态转移
        # states: [S_0, S_1, ..., S_T]
        # actions: [A_0, A_1, ..., A_T]
        # rewards: [R_0, R_1, ..., R_T]
        return states, actions, rewards

    def update(self, trajectory):  # 更新网络（带 baseline）
        """用轨迹数据更新策略网络（REINFORCE with baseline）"""
        states, actions, rewards = trajectory  # 解包
        # 逆序计算每个时间步的 G_t
        G = 0  # 初始化回报
        loss = 0  # 初始化损失
        for r, s, a in zip(rewards[::-1], states[::-1], actions[::-1]):  # 逆序遍历
            G = r + self.gamma * G  # 计算 G_t
            _, probs = self.get_action(s)  # 获取概率
            log_prob = torch.log(probs)[a]  # 对数概率
            # baseline: 减去常数基线 5.0，减少方差
            loss += - (G - 5.0) * log_prob  # 损失 = -(G_t - baseline) * log π

        self.optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数


env = gym.make("CartPole-v1")  # 创建环境
agent = Agent()  # 创建智能体
return_list = []  # 奖励记录
episode_list = []  # 回合记录

for episode in range(3000):  # 训练 3000 回合
    trajectory = agent.collect_trajectory(env)  # 收集轨迹
    agent.update(trajectory)  # 更新网络

    rewards = trajectory[2]  # 获取奖励
    return_list.append(sum(rewards))  # 总奖励
    episode_list.append(episode)  # 回合数

    if episode % 100 == 0:  # 每 100 回合打印
        print("回合：{}, 总奖励：{:.1f}".format(episode, sum(rewards)))

# 可视化
plot_loss(episode_list, return_list, "baseline-reinforce-pg-loss.pdf")  # 保存曲线


def test_agent(agent):  # 测试智能体
    """测试训练后的智能体"""
    # 测试时需要创建带渲染模式的环境
    test_env = gym.make("CartPole-v1", render_mode="rgb_array")  # 创建可渲染环境
    state, _ = test_env.reset()  # 重置
    done = False  # 结束标志
    frames = []  # 渲染帧

    while not done:  # 循环
        frames.append(test_env.render())  # 渲染
        action, _ = agent.get_action(state)  # 选择动作
        next_state, _, terminated, truncated, _ = test_env.step(action)  # 执行
        done = terminated or truncated  # 结束判断
        state = next_state  # 状态转移

    test_env.close()  # 关闭环境
    show_animation(frames)  # 显示动画


test_agent(agent)  # 测试
