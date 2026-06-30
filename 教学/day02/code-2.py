import gymnasium as gym  # 导入 Gymnasium 库（Gym 的维护版本）
import random  # 导入随机数库（本例中未使用）
import matplotlib.pyplot as plt  # 导入 matplotlib 绑图库
import matplotlib.animation as animation  # 导入动画库
from matplotlib import rc  # 导入 matplotlib 配置
import numpy as np  # 导入 NumPy 库（本例中未使用）
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式 API
import torch.optim as optim  # 导入优化器
from torch.distributions import Categorical  # 导入分类分布


def show_animation(imgs):  # 显示并保存动画的函数
    """将图片序列转为动画并保存"""
    rc("animation", html="jshtml")  # 设置 matplotlib 动画后端
    fig, ax = plt.subplots(1, 1, figsize=(5, 3))  # 创建图形和坐标轴
    frames = []  # 初始化帧列表

    text = ax.text(10, 20, "", fontsize=12, color="black")  # 创建文本对象（未使用）

    for i, img in enumerate(imgs):  # 遍历每张图片
        frame = [ax.imshow(img, animated=True)]  # 将图片作为动画帧
        frame.append(ax.text(10, 20, f"Step: {i+1}", animated=True))  # 添加步数标注
        frames.append(frame)  # 添加到帧列表

    ax.axis("off")  # 关闭坐标轴显示

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)  # 创建动画，间隔 100ms

    # 保存动画
    ani.save("cartpole.mp4", writer="ffmpeg")  # 保存为 MP4 格式
    ani.save("cartpole.gif", writer="pillow")  # 保存为 GIF 格式

    plt.close(fig)  # 关闭图形
    return ani  # 返回动画对象


def plot_loss(episode_list, return_list, filename):  # 绘制奖励曲线
    """绘制奖励图像"""
    f = plt.figure()  # 创建新图形
    plt.plot(episode_list, return_list)  # 绘制回合-奖励曲线
    plt.xlabel("Episodes")  # x 轴标签
    plt.ylabel("Returns")  # y 轴标签
    plt.title("CartPole-v0")  # 标题
    plt.show()  # 显示图形
    f.savefig(filename, bbox_inches="tight")  # 保存为 PDF 文件


class PolicyNet(nn.Module):  # 策略神经网络
    """策略神经网络的结构"""

    def __init__(self, action_size):  # 构造函数
        super().__init__()  # 调用父类构造函数
        self.l1 = nn.Linear(4, 128)  # 第一层：4 维状态 → 128 维隐藏层
        self.l2 = nn.Linear(128, action_size)  # 第二层：128 维 → 动作概率

    def forward(self, x):  # x 是 S_t（状态）
        x = F.relu(self.l1(x))  # ReLU 激活
        x = F.softmax(self.l2(x), dim=1)  # Softmax 输出概率分布
        return x


class Agent:  # 智能体类
    def __init__(self):  # 构造函数
        self.gamma = 0.98  # 折扣因子 γ
        self.lr = 0.0005  # 学习率 α
        self.action_size = 2  # 两个动作：向左推和向右推
        # 初始化策略网络 π_θ
        self.pi = PolicyNet(self.action_size)  # 创建策略网络
        self.optimizer = optim.Adam(  # 创建 Adam 优化器
            self.pi.parameters(),  # 优化网络参数
            lr=self.lr  # 设置学习率
        )

    def get_action(self, state):  # 根据状态选择动作
        """输入参数：环境的状态"""
        # 动作的概率分布
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)  # 状态 → 概率分布
        # 创建一个分类分布采样器
        m = Categorical(probs)  # 基于概率创建分布
        # 采样一个动作
        action = m.sample().item()  # 从分布中采样

        return action, probs  # 返回动作和概率

    def collect_trajectory(self, env):  # 收集一条完整轨迹
        """在环境 env 中采样一条轨迹"""
        state, _ = env.reset()  # 重置环境，获取初始状态（Gymnasium 新 API）
        states, actions, rewards = [], [], []  # 初始化轨迹列表
        done = False  # 结束标志

        while not done:  # 循环直到回合结束
            # 采取动作 A_t
            action, _ = self.get_action(state)  # 选择动作
            # 执行动作 A_t
            next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作
            done = terminated or truncated  # 判断是否结束

            states.append(state)  # 保存状态
            actions.append(action)  # 保存动作
            rewards.append(reward)  # 保存奖励

            # 状态转移
            state = next_state  # S_t → S_{t+1}
        # states: [S_0, S_1, S_2, ..., S_T]
        # actions: [A_0, A_1, A_2, ..., A_T]
        # rewards: [R_0, R_1, R_2, ..., R_T]
        return states, actions, rewards  # 返回轨迹

    def update(self, trajectory):  # 更新策略网络
        """用轨迹 trajectory 数据更新策略网络"""
        states, actions, rewards = trajectory  # 解包轨迹数据
        # 逆序计算 G(τ) - 整条轨迹的累积回报
        G = 0  # 初始化回报
        for r in rewards[::-1]:  # 逆序遍历奖励
            G = r + self.gamma * G  # 递推计算：G = R_t + γ * G_{t+1}
        # 计算损失
        states = torch.tensor(states)  # 状态转张量
        actions = torch.tensor(actions).view(-1, 1)  # 动作转张量，reshape 为列向量
        log_probs = torch.log(self.pi(states).gather(1, actions))  # 计算每个动作的对数概率
        loss = -torch.sum(log_probs) * G  # 策略梯度损失：-Σ log π(a|s) * G(τ)

        self.optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        self.optimizer.step()  # 更新参数


env = gym.make("CartPole-v1")  # 创建 CartPole 环境
agent = Agent()  # 创建智能体
return_list = []  # 记录每回合的总奖励
episode_list = []  # 记录回合编号

for episode in range(3000):  # 训练 3000 个回合
    # 采样一条轨迹
    trajectory = agent.collect_trajectory(env)  # 收集轨迹
    # 使用采样到的轨迹更新策略
    agent.update(trajectory)  # 更新网络

    # 统计数据
    rewards = trajectory[2]  # 获取奖励列表
    return_list.append(sum(rewards))  # 计算总奖励
    episode_list.append(episode)  # 记录回合数

    if episode % 100 == 0:  # 每 100 回合打印一次
        print("回合：{}, 总奖励：{:.1f}".format(episode, sum(rewards)))  # 输出训练信息

# 可视化损失
plot_loss(episode_list, return_list, "pg-loss.pdf")  # 绘制并保存奖励曲线


def test_agent(agent):  # 测试训练后的智能体
    """测试训练后的智能体"""
    # 测试时需要创建带渲染模式的环境
    test_env = gym.make("CartPole-v1", render_mode="rgb_array")  # 创建可渲染环境
    state, _ = test_env.reset()  # 重置环境
    done = False  # 结束标志
    frames = []  # 存储渲染帧

    while not done:  # 循环直到结束
        frames.append(test_env.render())  # 渲染当前帧
        action, _ = agent.get_action(state)  # 选择动作
        next_state, _, terminated, truncated, _ = test_env.step(action)  # 执行动作
        done = terminated or truncated  # 判断是否结束
        state = next_state  # 状态转移

    test_env.close()  # 关闭环境
    show_animation(frames)  # 显示动画


test_agent(agent)  # 测试智能体
