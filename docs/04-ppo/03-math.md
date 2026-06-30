# PPO 数学推导

## 为什么需要数学推导？

PPO（Proximal Policy Optimization）是目前最流行的强化学习算法之一。理解它的数学原理，能帮助我们：

1. **理解裁剪机制的本质** - 为什么 clip 能保证策略更新"安全"
2. **掌握重要性采样** - 如何重用旧数据提高样本效率
3. **建立理论直觉** - 为调参和改进算法打下基础

---

## 前置知识

### 1. 策略梯度回顾

策略梯度法的核心公式：

$$\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

其中 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是从时刻 $t$ 开始的累积回报。

**问题**：每次更新后，旧数据就不能用了，因为策略变了。

### 2. 为什么 PPO 能解决这个问题？

PPO 使用**重要性采样**（Importance Sampling）来重用旧数据，同时用**裁剪机制**（Clipping）保证策略不会变化太大。

---

## 重要性采样

### 基本思想

假设我们想计算 $E_{x \sim p}[f(x)]$，但只有从分布 $q$ 采样的数据。重要性采样告诉我们：

$$E_{x \sim p}[f(x)] = E_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$$

其中 $\frac{p(x)}{q(x)}$ 称为**重要性权重**（importance weight）。

### 直观理解

想象你在估计"喜欢吃辣的人的比例"，但你的样本都是四川人。你需要给每个样本一个权重：

- 四川人：权重低（因为他们本来就吃辣）
- 广东人：权重高（因为他们通常不吃辣）

这样加权后的平均值才能代表整体情况。

### 在策略梯度中的应用

我们想计算新策略 $\pi_\theta$ 的期望回报，但只有旧策略 $\pi_{\theta_{old}}$ 的数据：

$$E_{\tau \sim \pi_\theta}[R(\tau)] = E_{\tau \sim \pi_{\theta_{old}}}\left[\frac{P(\tau; \theta)}{P(\tau; \theta_{old})} R(\tau)\right]$$

其中：
- $P(\tau; \theta)$ 是在策略 $\pi_\theta$ 下产生轨迹 $\tau$ 的概率
- $P(\tau; \theta_{old})$ 是在旧策略下产生轨迹 $\tau$ 的概率

---

## 轨迹概率的分解

### 轨迹的定义

一条轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ...)$ 包含了状态、动作和奖励的序列。

### 轨迹概率公式

在策略 $\pi_\theta$ 下，轨迹 $\tau$ 的概率为：

$$P(\tau; \theta) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|s_t, a_t)$$

其中：
- $p(s_0)$：初始状态的概率（与策略无关）
- $\pi_\theta(a_t|s_t)$：在状态 $s_t$ 下选择动作 $a_t$ 的概率
- $p(s_{t+1}|s_t, a_t)$：状态转移概率（与策略无关，由环境决定）

### 取对数

对两边取对数，把乘积变成求和：

$$\log P(\tau; \theta) = \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{T} \log p(s_{t+1}|s_t, a_t)$$

**关键观察**：只有 $\sum_{t=0}^{T} \log \pi_\theta(a_t|s_t)$ 这一项依赖于 $\theta$，其他两项与 $\theta$ 无关。

---

## 重要性权重的简化

### 完整的重要性权重

$$\frac{P(\tau; \theta)}{P(\tau; \theta_{old})} = \frac{p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t|s_t) \cdot p(s_{t+1}|s_t, a_t)}{p(s_0) \prod_{t=0}^{T} \pi_{\theta_{old}}(a_t|s_t) \cdot p(s_{t+1}|s_t, a_t)}$$

### 简化

分子分母中的 $p(s_0)$ 和 $p(s_{t+1}|s_t, a_t)$ 相消：

$$\frac{P(\tau; \theta)}{P(\tau; \theta_{old})} = \prod_{t=0}^{T} \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

### 单步重要性权重

在实际应用中，我们通常使用**单步**的重要性权重：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

这是因为：
1. 计算更简单
2. 方差更小
3. 实践证明效果更好

---

## CPI 目标函数

### 定义

CPI（Conservative Policy Iteration）目标函数：

$$L^{CPI}(\theta) = E_{\tau \sim \pi_{\theta_{old}}}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A_t\right]$$

其中 $A_t = Q(s_t, a_t) - V(s_t)$ 是**优势函数**。

### 直观理解

- 如果 $A_t > 0$（好动作），我们想增加 $\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
- 如果 $A_t < 0$（坏动作），我们想减少 $\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

### 问题

无约束的优化可能导致 $\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 变得非常大或非常小，导致策略变化太大，训练不稳定。

---

## PPO 裁剪目标

### 核心思想

PPO 用**裁剪机制**限制 $r_t(\theta)$ 的范围，确保策略不会变化太大。

### 公式

$$L^{CLIP}(\theta) = E\left[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)\right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
- $\epsilon$ 是裁剪参数，通常取 0.1 或 0.2
- $\text{clip}(x, a, b)$ 把 $x$ 限制在 $[a, b]$ 范围内

### 分情况讨论

#### 情况 1：$A_t > 0$（好动作）

我们想增加 $r_t(\theta)$，但裁剪限制了 $r_t(\theta) \leq 1 + \epsilon$。

```
L^CLIP
    ^
    |     /|
    |    / |
    |   /  |
    |  /   |
    | /    |
----+-------> r(θ)
   1-ε    1    1+ε
```

- 当 $r_t(\theta) < 1 + \epsilon$ 时：$L^{CLIP} = r_t(\theta) A_t$（正常优化）
- 当 $r_t(\theta) \geq 1 + \epsilon$ 时：$L^{CLIP} = (1+\epsilon) A_t$（被裁剪，不再增加）

**效果**：防止过度增加好动作的概率，保持策略稳定。

#### 情况 2：$A_t < 0$（坏动作）

我们想减少 $r_t(\theta)$，但裁剪限制了 $r_t(\theta) \geq 1 - \epsilon$。

```
L^CLIP
    ^
    |\    
    | \   
    |  \  
    |   \ 
    |    \
----+-------> r(θ)
   1-ε    1    1+ε
```

- 当 $r_t(\theta) > 1 - \epsilon$ 时：$L^{CLIP} = r_t(\theta) A_t$（正常优化）
- 当 $r_t(\theta) \leq 1 - \epsilon$ 时：$L^{CLIP} = (1-\epsilon) A_t$（被裁剪，不再减少）

**效果**：防止过度减少坏动作的概率，保持策略稳定。

### 为什么是 min 操作？

$$L^{CLIP} = \min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)$$

- 当 $A_t > 0$ 时：$\min$ 选择较小的值，即 $(1+\epsilon)A_t$（上限）
- 当 $A_t < 0$ 时：$\min$ 选择较小的值，即 $r_t(\theta)A_t$（下限）

**统一效果**：无论 $A_t$ 正负，都限制了策略变化的幅度。

---

## 全变差距离

### 定义

全变差距离（Total Variation Distance）衡量两个分布的差异：

$$D_{TV}(P || Q) = \frac{1}{2} \sum_x |P(x) - Q(x)|$$

### 与 KL 散度的关系

$$D_{TV}(P || Q)^2 \leq KL(P || Q) \leq 2 D_{TV}(P || Q)^2$$

### 直观理解

- $D_{TV}$ 衡量"两个分布有多不同"
- $KL$ 散度衡量"用 $Q$ 近似 $P$ 会损失多少信息"
- 两者都能量化策略变化的程度

---

## 误差上界

### 策略更新的误差

$$|J(\theta) - J(\theta_{old})| \leq C \cdot D_{TV}(\pi_\theta || \pi_{\theta_{old}})$$

其中 $C$ 是一个常数。

### 直观理解

- 如果策略变化太大（$D_{TV}$ 大），新策略的表现可能比旧策略差很多
- PPO 的裁剪机制限制了 $r_t(\theta)$ 的范围，从而限制了 $D_{TV}$

### PPO 的理论保证

裁剪机制确保：
1. $r_t(\theta) \in [1-\epsilon, 1+\epsilon]$
2. 策略变化在可控范围内
3. 新策略的表现不会比旧策略差太多

---

## 完整的 PPO 目标函数

### 公式

$$L(\theta) = E\left[L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s)\right]$$

其中：
- $L^{CLIP}(\theta)$：裁剪的策略目标（上面推导的）
- $L^{VF}(\theta)$：价值函数损失（通常用 MSE）
- $S[\pi_\theta](s)$：熵正则化项（鼓励探索）
- $c_1, c_2$：系数，平衡各项的重要性

### 为什么需要价值函数损失？

- 价值函数 $V(s)$ 用于计算优势函数 $A_t = Q(s_t, a_t) - V(s_t)$
- 如果 $V(s)$ 不准，$A_t$ 也不准，策略更新就会有问题
- 最小化 $L^{VF}$ 确保 $V(s)$ 尽可能准确

### 为什么需要熵正则化？

- 熵 $S[\pi_\theta](s) = -\sum_a \pi_\theta(a|s) \log \pi_\theta(a|s)$ 衡量策略的随机性
- 熵越大，策略越随机，探索越多
- 防止策略过早收敛到局部最优

---

## 代码实现

### PPO 损失函数

```python
import torch
import torch.nn as nn

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

### 训练循环

```python
def train_ppo(policy, optimizer, states, actions, old_log_probs,
              returns, advantages, clip_epsilon=0.2, n_epochs=10):
    """
    PPO 训练循环

    对同一批数据训练 n_epochs 轮
    """
    for _ in range(n_epochs):
        # 前向传播
        new_log_probs, values = policy.evaluate(states, actions)

        # 计算损失
        loss, actor_loss, critic_loss, entropy_loss = ppo_loss(
            old_log_probs, new_log_probs, advantages, returns, values,
            clip_epsilon=clip_epsilon
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()
```

---

## 关键要点

### 1. 重要性采样

- 允许重用旧策略的数据
- 重要性权重 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
- 方差大是主要问题，需要裁剪

### 2. 裁剪机制

- 限制 $r_t(\theta) \in [1-\epsilon, 1+\epsilon]$
- 防止策略变化太大
- 保证训练稳定

### 3. 三重损失

- **策略损失** $L^{CLIP}$：优化策略
- **价值损失** $L^{VF}$：学习价值函数
- **熵损失** $S$：鼓励探索

### 4. 超参数选择

| 参数 | 推荐值 | 作用 |
|------|--------|------|
| $\epsilon$ | 0.1-0.2 | 裁剪范围 |
| $c_1$ | 0.5 | 价值损失权重 |
| $c_2$ | 0.01 | 熵损失权重 |
| $n\_epochs$ | 3-10 | 每批数据训练轮数 |

---

## 延伸阅读

1. **TRPO**（Trust Region Policy Optimization）：PPO 的前身，用 KL 散度约束
2. **PPO2**：OpenAI 的改进版本，用 GAE 计算优势函数
3. **PPO-Clip** vs **PPO-Penalty**：两种不同的约束方式

---

## 总结

PPO 的数学核心是：

1. **重要性采样**：重用旧数据
2. **裁剪机制**：限制策略变化
3. **三重损失**：平衡策略、价值、探索

理解这些数学原理，能帮助你：
- 更好地调参
- 理解训练过程中的问题
- 改进或设计新的算法
