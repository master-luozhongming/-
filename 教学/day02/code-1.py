import gymnasium as gym  # 导入 Gymnasium 库（Gym 的维护版本）
import random  # 导入随机数库（本例中未使用）
import matplotlib.pyplot as plt  # 导入绑图库（本例中未使用）
import matplotlib.animation as animation  # 导入动画库（本例中未使用）
from matplotlib import rc  # 导入 matplotlib 配置（本例中未使用）
import numpy as np  # 导入 NumPy 库（本例中未使用）
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块
import torch.nn.functional as F  # 导入函数式 API（relu、softmax 等）
import torch.optim as optim  # 导入优化器
from torch.distributions import Categorical  # 导入分类分布，用于从概率分布中采样


# 神经网络表达策略函数 π(a|s)
class PolicyNet(nn.Module):  # 定义策略网络，继承自 nn.Module
    def __init__(self, action_size):  # 构造函数，action_size 为动作数量
        super().__init__()  # 调用父类构造函数
        self.l1 = nn.Linear(4, 128)  # 第一层：输入 4 维状态，输出 128 维
        self.l2 = nn.Linear(128, action_size)  # 第二层：输入 128 维，输出动作概率

    def forward(self, x):  # 前向传播
        x = F.relu(self.l1(x))  # 第一层后使用 ReLU 激活函数
        x = F.softmax(self.l2(x), dim=1)  # 第二层后使用 Softmax 输出概率分布
        return x  # 返回动作概率


class Agent:  # 定义智能体类
    def __init__(self):  # 构造函数
        self.gamma = 0.98  # 折扣因子
        self.lr = 0.0002  # 学习率
        self.action_size = 2  # 动作数量（左移、右移）
        # 初始化策略网络 π_θ
        self.pi = PolicyNet(self.action_size)  # 创建策略网络实例
        self.optimizer = optim.Adam(  # 创建 Adam 优化器
            self.pi.parameters(),  # 优化策略网络的参数
            lr=self.lr  # 设置学习率
        )

    def get_action(self, state):  # 根据状态选择动作
        """输入参数：环境的状态"""
        probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0)  # 状态转张量，输入网络得到概率
        m = Categorical(probs)  # 创建分类分布
        action = m.sample().item()  # 从分布中采样动作

        return action, probs  # 返回动作和概率分布


env = gym.make("CartPole-v1")  # 创建 CartPole-v1 环境
state, _ = env.reset()  # 重置环境，获取初始状态 S₀（Gymnasium 新 API 返回元组）
agent = Agent()  # 创建智能体实例

action, probs = agent.get_action(state)  # 获取动作和概率分布
print("采取的动作：", "向左推" if action == 0 else "向右推")  # 打印动作名称
print("采取的动作的概率：", probs[action].item())  # 打印该动作的概率

G = 1.0  # 假设的回报 G(τ)
J = -G * probs[action].log()  # 计算策略梯度损失：−𝐺(𝜏)log𝜋𝜃(𝐴0|𝑆0)
J.backward()  # 反向传播计算梯度
agent.optimizer.step()  # 更新网络参数

_, probs = agent.get_action(state)  # 更新后再次获取概率分布
print("采取的动作：", "向左推" if action == 0 else "向右推")  # 打印动作名称
print("采取的动作的概率：", probs[action].item())  # 打印更新后的动作概率（应该变大）
