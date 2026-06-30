# 强化学习与大模型教程 - 知识蒸馏

> 基于左元《大语言模型、强化学习和多模态教程》整理

## 📚 目录结构

```
reinforcement/
├── README.md                 # 本文件 - 知识蒸馏总览
├── 01_LLM/                   # 大语言模型部分
│   ├── microgpt.md          # microgpt实现详解
│   ├── llm_intro.md         # LLM基础概念
│   └── gpt2.md              # GPT-2实现
├── 02_RL_Basics/             # 强化学习基础
│   ├── concepts.md          # 核心概念
│   ├── value_function.md    # 价值函数
│   └── bellman_equation.md  # 贝尔曼方程
├── 03_Policy_Gradient/       # 策略梯度法
│   ├── vanilla_pg.md        # 原始策略梯度法
│   ├── reinforce.md         # REINFORCE算法
│   └── actor_critic.md      # Actor-Critic架构
├── 04_PPO/                   # 近端策略优化
│   ├── theory.md            # PPO原理
│   └── implementation.md    # PPO实战
├── 05_RLHF/                  # 基于人类反馈的强化学习
│   ├── dpo.md               # 直接偏好优化
│   └── grpo.md              # 组相对策略优化
├── 06_Multimodal/            # 多模态
│   ├── vit.md               # Vision Transformer
│   ├── clip.md              # CLIP模型
│   └── diffusion.md         # 扩散模型
└── code/                     # 代码示例
    ├── rl_basics/
    ├── policy_gradient/
    ├── ppo/
    └── grpo/
```

---

## 🎯 核心概念速查

### 强化学习基本要素

| 概念 | 符号 | 含义 |
|------|------|------|
| 智能体 | Agent | 学习和决策的主体 |
| 环境 | Environment | 智能体交互的对象 |
| 状态 | State (s) | 环境的当前状态 |
| 动作 | Action (a) | 智能体采取的行为 |
| 奖励 | Reward (r) | 即时反馈信号 |
| 策略 | π(a\|s) | 状态到动作的映射 |
| 回报 | G_t | 累积折扣奖励 |

### 关键公式

**回报 (Return)**
```
G_t = R_t + γR_{t+1} + γ²R_{t+2} + ...
    = R_t + γG_{t+1}
```

**状态价值函数**
```
V_π(s) = E_π[G_t | S_t = s]
```

**贝尔曼期望方程**
```
V_π(S_t) = E_π[R_t + γV_π(S_{t+1}) | S_t]
```

---

## 📖 章节详解

### Part I: 大语言模型

#### 1. microgpt
- 数据集准备
- 分词器 (Tokenizer)
- 自动微分 (Autograd)
- 模型架构
- 训练循环

#### 2. LLM 简介
- 大语言模型要解决的问题
- 如何训练预测下一个token的模型
- 训练数据组织方式
- 模型结构设计
- 损失函数设计

#### 3. GPT-2
- GPT-2模型结构定义
- 数据集准备
- 模型训练
- 加载OpenAI预训练权重

---

### Part II: 强化学习

#### 4. 强化学习简介
- **基本概念**: Agent, Environment, State, Reward, Action
- **价值函数**: 状态价值函数V(s), 动作价值函数Q(s,a)
- **贝尔曼方程**: 动态规划的基础
- **倒立摆环境**: 经典控制问题

#### 5. 策略梯度法
- **原始策略梯度法**: ∇J(θ) = E[∇log π_θ(a|s) · G_t]
- **REINFORCE**: 蒙特卡洛策略梯度
- **带基线的策略梯度**: 减少方差
- **Actor-Critic**: 演员-评论家架构
- **广义优势估计 (GAE)**: 平衡偏差及方差

#### 6. 近端策略优化 (PPO)
- 策略梯度法存在的问题
- 置信域方法
- PPO目标函数
- 裁剪机制

#### 7. PPO 实战
- 网络结构定义
- PPO算法实现
- 广义优势估计计算
- 训练循环

#### 8. PPO 背后的数学
- KL散度
- 重要性采样
- 替代目标推导

#### 9. 组相对策略优化 (GRPO)
- GRPO原理
- 使用GRPO玩倒立摆游戏

---

### Part III: RLHF

#### 10. 大语言模型训练概述
- LLM训练流程: 预训练 → SFT → RLHF
- PPO在RLHF中的应用
- DPO直接偏好优化
- GRPO组相对策略优化

#### 11. DPO
- DPO原理
- 使用DPO微调大语言模型

#### 12. GRPO进阶
- GRPO数学推导
- GRPO应用场景
- 医疗思维链
- Text-To-SQL

---

### Part IV: 多模态

#### 14. Vision Transformer
- 补丁嵌入 (Patch Embedding)
- 位置编码
- 注意力机制
- Transformer编码器

#### 15. CLIP
- 图像和文本编码器
- 对比学习

#### 16. 扩散模型
- 扩散过程
- 反向扩散
- U-Net模型
- 条件扩散模型

---

## 💻 代码示例

### 倒立摆环境基础

```python
import gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# 创建环境
env = gym.make("CartPole-v0")

# 获取初始状态
state = env.reset()
print(f"初始状态: {state}")  # [位置, 速度, 角度, 角速度]

# 动作空间
print(f"动作空间: {env.action_space}")  # Discrete(2)

# 执行一步
action = 0  # 0=向左, 1=向右
next_state, reward, done, info = env.step(action)
```

### 策略网络

```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)

    def get_action(self, state):
        probs = self.forward(torch.FloatTensor(state))
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
```

### REINFORCE 算法

```python
def reinforce(policy, optimizer, gamma=0.99):
    # 收集一条轨迹
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False

    while not done:
        action, log_prob = policy.get_action(state)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    # 计算回报
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)

    # 计算损失
    loss = 0
    for log_prob, G in zip(log_probs, returns):
        loss -= log_prob * G

    # 更新策略
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Actor-Critic

```python
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Linear(state_dim, 128)
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.shared(x))
        action_probs = torch.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return action_probs, value

    def get_action(self, state):
        probs, value = self.forward(torch.FloatTensor(state))
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
```

### PPO 核心实现

```python
def ppo_update(policy, states, actions, old_log_probs, returns, advantages,
               clip_epsilon=0.2, epochs=4):
    for _ in range(epochs):
        # 计算新的log概率
        new_log_probs, values = policy.evaluate(states, actions)

        # 计算比率
        ratio = torch.exp(new_log_probs - old_log_probs)

        # 裁剪目标
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

        # PPO损失
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = nn.MSELoss()(values, returns)
        entropy_loss = -dist.entropy().mean()

        loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 广义优势估计 (GAE)

```python
def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return torch.FloatTensor(advantages)
```

---

## 🔑 关键算法对比

| 算法 | 类型 | 优点 | 缺点 |
|------|------|------|------|
| REINFORCE | 策略梯度 | 简单直观 | 方差大，收敛慢 |
| Actor-Critic | 策略梯度+价值函数 | 方差较小 | 需要调参 |
| PPO | 策略梯度 | 稳定，样本效率高 | 计算开销大 |
| DPO | 偏好优化 | 无需奖励模型 | 需要偏好数据 |
| GRPO | 组相对优化 | 适合LLM | 新方法 |

---

## 📊 数学推导要点

### 策略梯度定理
```
∇J(θ) = E_τ[∑_t ∇log π_θ(a_t|s_t) · G_t]
```

### PPO目标函数
```
L(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```
其中 r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)

### DPO损失函数
```
L(θ) = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

---

## 🛠️ 环境配置

```bash
# 基础依赖
pip install gym==0.25.2
pip install numpy==1.26
pip install pygame
pip install torch
pip install matplotlib

# 可选
pip install tensorboard  # 训练可视化
```

---

## 📚 参考资料

- 原始文档: 《大语言模型、强化学习和多模态教程》- 左元
- OpenAI Gym: https://gym.openai.com
- PyTorch: https://pytorch.org
- Spinning Up in Deep RL: https://spinningup.openai.com

---

## 📝 学习路径建议

1. **入门**: 强化学习基本概念 → 倒立摆环境
2. **基础**: 策略梯度法 → REINFORCE
3. **进阶**: Actor-Critic → PPO
4. **应用**: DPO/GRPO → LLM微调
5. **扩展**: 多模态 → ViT/CLIP/扩散模型

---

*Last updated: 2024*
