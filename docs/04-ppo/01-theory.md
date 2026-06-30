# PPO 理论

## 为什么需要 PPO？

在强化学习中，策略梯度法（如 REINFORCE）存在一个严重问题：**步幅选择困难**。

- 步幅太大 → 策略崩溃，性能急剧下降
- 步幅太小 → 收敛太慢，训练效率低

PPO（Proximal Policy Optimization）就是为了解决这个问题而设计的。

---

## 核心思想

### 置信域方法

PPO 的核心思想来自**置信域方法**（Trust Region Methods）：

> 限制新策略与旧策略的差异，确保更新在"置信域"内。

### 直观理解

想象你在爬山：
- **传统策略梯度**：每次走一大步，可能走到悬崖
- **PPO**：每次走一小步，确保不会走太远

---

## 策略梯度法回顾

### 基本公式

策略梯度法的目标函数：

$$J(\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

梯度：

$$\nabla_\theta J(\theta) = E_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t \right]$$

### 问题

每次更新后，旧数据就不能用了，因为策略变了。这导致：

1. **样本效率低**：每条轨迹只用一次
2. **训练不稳定**：小的参数变化可能导致策略剧烈变化

---

## 重要性采样

### 基本思想

重要性采样允许我们用旧策略的数据来估计新策略的期望：

$$E_{x \sim p}[f(x)] = E_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$$

### 在策略梯度中的应用

我们想计算新策略 $\pi_\theta$ 的期望回报，但只有旧策略 $\pi_{\theta_{old}}$ 的数据：

$$E_{\tau \sim \pi_\theta}[R(\tau)] = E_{\tau \sim \pi_{\theta_{old}}}\left[\frac{P(\tau; \theta)}{P(\tau; \theta_{old})} R(\tau)\right]$$

### 重要性权重

$$\frac{P(\tau; \theta)}{P(\tau; \theta_{old})} = \prod_{t=0}^{T} \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

在实际应用中，我们使用**单步**的重要性权重：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

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
L^{CLIP}
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
L^{CLIP}
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

## 深入讨论

### 为什么 PPO 比 TRPO 更受欢迎？

TRPO（Trust Region Policy Optimization）是 PPO 的前身，使用 KL 散度约束：

$$\max_\theta E\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A(s,a)\right]$$
$$\text{s.t. } KL(\pi_{\theta_{old}} || \pi_\theta) \leq \delta$$

PPO 的优势：

| 特性 | TRPO | PPO |
|------|------|-----|
| 实现复杂度 | 高（需要二阶优化） | 低（只需要一阶优化） |
| 计算开销 | 大 | 小 |
| 调参难度 | 难 | 容易 |
| 效果 | 好 | 好 |

### 裁剪参数 $\epsilon$ 的选择

- $\epsilon$ 太小（如 0.05）：策略更新太保守，收敛慢
- $\epsilon$ 太大（如 0.3）：策略更新太激进，可能不稳定
- 推荐值：0.1-0.2

### 多轮更新

PPO 可以对同一批数据进行多轮更新（通常 3-10 轮），这是 REINFORCE 做不到的。

**优势**：
- 提高样本效率
- 更充分地利用数据

**注意**：
- 轮数太多可能导致过拟合
- 通常需要监控 KL 散度，确保策略没有变化太大

---

## 关键要点

1. **PPO 是最流行的策略梯度算法之一**
2. **裁剪机制**确保更新在安全范围内
3. **重要性采样**允许重用旧数据
4. **三重损失**平衡策略、价值、探索
5. **广泛应用于**游戏、机器人、LLM 等领域

---

## 延伸阅读

1. **TRPO**：PPO 的前身，用 KL 散度约束
2. **PPO2**：OpenAI 的改进版本，用 GAE 计算优势函数
3. **PPO-Clip** vs **PPO-Penalty**：两种不同的约束方式
