# 广义优势估计 (GAE)

## 为什么需要 GAE？

### 偏差-方差权衡

| 方法 | 偏差 | 方差 | 说明 |
|------|------|------|------|
| 蒙特卡洛 $G_t$ | 无 | 高 | 完整轨迹回报 |
| 单步 TD $\delta_t$ | 有 | 低 | $r_t + \gamma V(s_{t+1}) - V(s_t)$ |
| n-step TD | 中 | 中 | 折中方案 |

GAE 通过参数 $\lambda$ 平滑地平衡偏差和方差。

---

## n-step 回报

### 定义

$$G_t^{(n)} = r_t + \gamma r_{t+1} + ... + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})$$

### 例子

- $n=1$: $G_t^{(1)} = r_t + \gamma V(s_{t+1})$ （TD 目标）
- $n=2$: $G_t^{(2)} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2})$
- $n=\infty$: $G_t^{(\infty)} = G_t$ （蒙特卡洛回报）

---

## GAE 公式

### 核心思想

GAE 是所有 n-step 回报的指数加权平均：

$$A_t^{GAE(\gamma, \lambda)} = (1-\lambda)(A_t^{(1)} + \lambda A_t^{(2)} + \lambda^2 A_t^{(3)} + ...)$$

### 简化公式

$$A_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 误差。

### 递推计算

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    计算广义优势估计

    A_t = ∑_{l=0}^∞ (γλ)^l δ_{t+l}
    """
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        # TD 误差
        delta = rewards[t] + gamma * next_values[t] * (1 - dones[t]) - values[t]

        # GAE 递推
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return torch.FloatTensor(advantages)
```

---

## 参数 λ 的影响

### λ = 0

$$A_t^{GAE(\gamma, 0)} = \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

退化为单步 TD，**偏差大，方差小**。

### λ = 1

$$A_t^{GAE(\gamma, 1)} = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l} = G_t - V(s_t)$$

退化为蒙特卡洛优势，**无偏，方差大**。

### λ ∈ (0, 1)

在偏差和方差之间取得平衡。

---

## 完整实现

```python
import torch
import numpy as np

def compute_gae_and_returns(rewards, values, next_values, dones,
                            gamma=0.99, lam=0.95):
    """
    计算 GAE 和回报

    Args:
        rewards: 奖励序列
        values: 当前状态价值
        next_values: 下一状态价值
        dones: 是否结束标志
        gamma: 折扣因子
        lam: GAE 参数

    Returns:
        advantages: 优势估计
        returns: 回报
    """
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0 if dones[t] else next_values[t]
        else:
            next_value = values[t + 1]

        # TD 误差
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

        # GAE
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values)

    # 标准化优势
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns
```

---

## 在 PPO 中的应用

```python
def ppo_update(policy, states, actions, old_log_probs,
               advantages, returns, clip_epsilon=0.2):
    """PPO 更新"""
    # 计算新的 log 概率
    new_log_probs, values, entropy = policy.evaluate(states, actions)

    # 计算比率
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 裁剪目标
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    # 价值损失
    critic_loss = nn.MSELoss()(values, returns)

    # 熵损失
    entropy_loss = -entropy.mean()

    return actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss
```

---

## 关键要点

1. **GAE 平衡偏差和方差**
2. **λ 控制偏差-方差权衡**
3. **递推计算高效**
4. **是 PPO 的核心组件**
