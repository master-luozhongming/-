import gymnasium as gym  # 导入 Gymnasium 库，用于创建和管理强化学习环境
import random  # 导入随机数库（本例中未使用）
import matplotlib.pyplot as plt  # 导入 matplotlib 绑图库（本例中未使用）
import matplotlib.animation as animation  # 导入动画库（本例中未使用）
from matplotlib import rc  # 导入 matplotlib 配置（本例中未使用）
import numpy as np  # 导入 NumPy 库，用于数值计算（本例中未使用）
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入 PyTorch 神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 函数式 API（如 relu、softmax）
import torch.optim as optim  # 导入 PyTorch 优化器
from torch.distributions import Categorical  # 导入分类分布，用于从概率分布中采样


# 策略神经网络
class PolicyNet(nn.Module):  # 定义策略网络类，继承自 nn.Module
    def __init__(self, action_size):  # 构造函数，action_size 为动作空间大小
        super().__init__()  # 调用父类构造函数
        self.l1 = nn.Linear(4, 128)  # 第一层全连接层，输入维度 4（状态维度），输出维度 128
        self.l2 = nn.Linear(128, action_size)  # 第二层全连接层，输入维度 128，输出维度为动作数

    def forward(self, x):  # 前向传播函数，x 为输入状态
        x = F.relu(self.l1(x))  # 经过第一层后使用 ReLU 激活函数
        x = F.softmax(self.l2(x), dim=1)  # 经过第二层后使用 Softmax 输出动作概率分布
        return x  # 返回动作概率


# agent
class Agent:  # 定义智能体类
    def __init__(self):  # 构造函数
        self.gamma = 0.98  # 折扣因子，用于计算未来奖励的衰减
        self.lr = 0.0002   # 学习率，控制参数更新步长
        self.action_size = 2  # 动作数量（CartPole 有左移和右移两个动作）

        self.pi = PolicyNet(self.action_size)  # 初始化策略网络
        self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr)  # 创建 Adam 优化器

    def get_action(self, state):  # 根据状态选择动作
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)  # 将状态转为张量，输入网络得到动作概率
        # unsqueeze(0) 增加 batch 维度，squeeze(0) 去掉 batch 维度
        m = Categorical(probs)  # 根据动作的概率分布创建一个分类分布
        action = m.sample().item()  # 从分布中采样一个动作，并转为 Python 数字
        return action, probs  # 返回动作和对应的概率分布

    def collect_trajectory(self, env):  # 收集一条完整轨迹
        state, _ = env.reset()  # 重置环境，获取初始状态（Gymnasium 新 API 返回元组）
        states, actions, rewards = [], [], []  # 初始化状态、动作、奖励列表
        done = False  # 初始化结束标志

        while not done:  # 循环直到回合结束
            action, _ = self.get_action(state)  # 根据当前状态选择动作
            next_state, reward, terminated, truncated, _ = env.step(action)  # 执行动作，获取下一个状态和奖励
            done = terminated or truncated  # 判断回合是否结束（终止或截断）
            states.append(state)     # 保存当前状态 𝑆𝑡
            actions.append(action)   # 保存当前动作 𝐴𝑡
            rewards.append(reward)   # 保存当前奖励 𝑅𝑡
            state = next_state       # 状态转移 𝑆𝑡 → 𝑆𝑡+1

        return states, actions, rewards  # 返回轨迹中的所有状态、动作和奖励


env = gym.make("CartPole-v1")  # 创建 CartPole-v1 环境
state, _ = env.reset()  # 重置环境，获取初始状态 𝑆0
agent = Agent()  # 创建智能体实例

action, probs = agent.get_action(state)  # 根据初始状态获取动作和概率分布
print("动作：", action)  # 打印选择的动作 𝐴0
print("动作的概率：", probs[action].item())  # 打印该动作的概率 𝜋𝜃(𝐴0|𝑆0)

G = 100.0  # 假设的回报 𝐺(𝜏)
J = -G * probs[action].log()  # 计算策略梯度损失：−𝐺(𝜏)log 𝜋𝜃(𝐴0|𝑆0)
print("J: ", J)  # 打印损失值

J.backward()  # 反向传播，计算梯度 −∇𝜃𝐺(𝜏)log 𝜋𝜃(𝐴0|𝑆0)
agent.optimizer.step()  # 梯度下降更新参数：𝜃 = 𝜃 + 𝛼∇𝜃𝐺(𝜏)log 𝜋𝜃(𝐴0|𝑆0)

# 在相同的状态state下，采取动作的概率变大了
# （测试一下）将G变为负值，也就是负的奖励，会发现采取动作的概率下降了
_, probs = agent.get_action(state)  # 再次获取动作概率分布
print("动作：", action)  # 打印动作（未改变）
print("动作的概率：", probs[action].item())  # 打印更新后的动作概率，应该比之前更大
