# 近端策略优化 (PPO) 理论

## 策略梯度法存在的问题

### 问题1：步幅选择困难

```
∇J(θ) = E[∇log π_θ(a|s) · G_t]
```

- **步幅太大**: 策略直接崩溃
- **步幅太小**: 收敛太慢
- **方向不对**: 可能导致更差的策略

### 问题2：样本效率低

- 每条轨迹只用一次
- 无法重用旧数据

### 问题3：不稳定

- 小的参数变化可能导致策略剧烈变化
- 难以控制更新幅度

---

## 置信域方法 (Trust Region)

### 核心思想

在每一步更新中，限制新策略与旧策略的差异，确保更新在"置信域"内。

### 数学表达

```
maximize E[π_θ(a|s)/π_{θ_old}(a|s) · A(s, a)]
subject to: KL(π_{θ_old} || π_θ) ≤ δ
```

其中：
- π_{θ_old}: 旧策略
- π_θ: 新策略
- A(s, a): 优势函数
- δ: 置信域大小

---

## PPO 核心思想

### 1. 比率 (Ratio)

```
r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
```

- r_t(θ) = 1: 新旧策略相同
- r_t(θ) > 1: 新策略更倾向于选择该动作
- r_t(θ) < 1: 新策略更不倾向于选择该动作

### 2. 替代目标 (Surrogate Objective)

```
L^{CPI}(θ) = E[r_t(θ) · A_t]
```

这是原始策略梯度目标的替代形式。

### 3. 裁剪目标 (Clipped Objective)

```
L^{CLIP}(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```

其中 ε 是裁剪参数（通常为0.1或0.2）。

---

## PPO 目标函数详解

### 裁剪机制

```python
def clipped_surrogate(ratio, advantage, epsilon=0.2):
    # 未裁剪的目标
    surr1 = ratio * advantage

    # 裁剪后的目标
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage

    # 取最小值
    return torch.min(surr1, surr2)
```

### 直观理解

```
当 A > 0 (好动作):
    - ratio 被限制在 [1-ε, 1+ε] 范围内
    - 防止过度增加好动作的概率

当 A < 0 (坏动作):
    - ratio 被限制在 [1-ε, 1+ε] 范围内
    - 防止过度减少坏动作的概率
```

### 图示

```
       L^{CLIP}
           ^
           |     /|
           |    / |
           |   /  |
           |  /   |
           | /    |
    -------+-------> r(θ)
   1-ε    1      1+ε
```

---

## PPO 完整目标函数

### Actor-Critic 架构下的 PPO

```
L(θ) = E[L^{CLIP}(θ) - c_1 · L^{VF}(θ) + c_2 · S[π_θ](s)]
```

其中：
- L^{CLIP}(θ): 裁剪的策略目标
- L^{VF}(θ): 价值函数损失
- S[π_θ](s): 熵正则化项
- c_1, c_2: 系数

### 代码实现

```python
def ppo_loss(old_log_probs, new_log_probs, advantages, returns, values,
             clip_epsilon=0.2, c1=0.5, c2=0.01):
    # 计算比率
    ratio = torch.exp(new_log_probs - old_log_probs)

    # 裁剪目标
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()

    # 价值函数损失
    critic_loss = nn.MSELoss()(values, returns)

    # 熵损失（鼓励探索）
    entropy = -torch.sum(torch.exp(new_log_probs) * new_log_probs, dim=-1)
    entropy_loss = -entropy.mean()

    # 总损失
    total_loss = actor_loss + c1 * critic_loss + c2 * entropy_loss

    return total_loss, actor_loss, critic_loss, entropy_loss
```

---

## 广义优势估计 (GAE)

### 为什么需要 GAE

- 单步 TD 误差：方差小，偏差大
- 蒙特卡洛回报：方差大，偏差小
- GAE：平衡偏差和方差

### TD 误差

```
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

### GAE 公式

```
A_t^{GAE(γ,λ)} = ∑_{l=0}^∞ (γλ)^l δ_{t+l}
```

其中：
- γ: 折扣因子
- λ: GAE参数（控制偏差-方差权衡）

### 代码实现

```python
def compute_gae(rewards, values, next_values, dones, gamma=0.99, lam=0.95):
    """
    计算广义优势估计

    Args:
        rewards: 奖励列表
        values: 当前状态价值
        next_values: 下一状态价值
        dones: 是否结束标志
        gamma: 折扣因子
        lam: GAE参数

    Returns:
        advantages: 优势估计
        returns: 回报
    """
    advantages = []
    gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        # TD误差
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

## PPO 算法流程

### 算法伪代码

```
初始化策略参数 θ, 价值函数参数 φ
for iteration = 1, 2, ... do
    使用当前策略 π_θ 收集 N 条轨迹
    计算每个时刻的优势估计 A_t
    计算每个时刻的回报 R_t

    for epoch = 1, 2, ..., K do
        对收集的数据进行随机小批量采样
        计算策略比率 r_t(θ)
        计算裁剪目标 L^{CLIP}
        计算价值函数损失 L^{VF}
        计算熵损失 S
        计算总损失 L = L^{CLIP} - c1*L^{VF} + c2*S

        更新 θ ← θ - α∇L
    end for
end for
```

---

## PPO 超参数

### 关键超参数

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| clip_epsilon | 0.1-0.2 | 裁剪范围 |
| gamma | 0.99 | 折扣因子 |
| lam (GAE) | 0.95 | GAE参数 |
| learning_rate | 1e-4 - 3e-4 | 学习率 |
| n_epochs | 3-10 | 每次采集后的训练轮数 |
| batch_size | 64-256 | 小批量大小 |
| c1 (value loss) | 0.5 | 价值损失系数 |
| c2 (entropy) | 0.01 | 熵损失系数 |

### 调参建议

1. **clip_epsilon**: 过大导致不稳定，过小导致更新太慢
2. **n_epochs**: 过大导致过拟合，过小导致样本浪费
3. **lam**: λ=0 退化为单步TD，λ=1 退化为蒙特卡洛

---

## PPO 的优势

### 相比 REINFORCE
1. **样本效率高**: 数据可以重复使用
2. **稳定性好**: 裁剪机制限制更新幅度
3. **方差低**: 使用优势函数

### 相比 TRPO
1. **实现简单**: 不需要计算KL散度约束
2. **计算高效**: 只需要简单的裁剪操作
3. **效果相当**: 性能与TRPO相当

---

## 数学推导

### 策略梯度定理

```
∇J(θ) = E_τ[∑_t ∇log π_θ(a_t|s_t) · A_t]
```

### PPO 目标函数

```
L(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```

### 梯度

```
∇L(θ) = E[∇min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
```

---

## 关键要点

1. **PPO 是最流行的策略梯度算法之一**
2. **裁剪机制确保更新在安全范围内**
3. **GAE 平衡偏差和方差**
4. **广泛应用于游戏、机器人、LLM等领域**

---

## 与其他算法对比

| 算法 | 复杂度 | 稳定性 | 样本效率 | 适用场景 |
|------|--------|--------|----------|----------|
| REINFORCE | 低 | 低 | 低 | 简单任务 |
| A2C | 中 | 中 | 中 | 通用 |
| PPO | 中 | 高 | 高 | 通用 |
| TRPO | 高 | 高 | 高 | 需要严格约束 |

---

*参考: 原文档 Chapter 6, 8*
