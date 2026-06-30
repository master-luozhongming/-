# PPO 理论

## 策略梯度法的问题

### 问题 1：步幅选择困难

- 步幅太大 → 策略崩溃
- 步幅太小 → 收敛太慢

### 问题 2：样本效率低

每条轨迹只用一次，无法重用旧数据。

### 问题 3：不稳定

小的参数变化可能导致策略剧烈变化。

---

## 置信域方法

### 核心思想

限制新策略与旧策略的差异，确保更新在"置信域"内。

### 数学表达

$$\max_\theta E\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_{old}}(a|s)} A(s,a)\right]$$
$$\text{s.t. } KL(\pi_{\theta_{old}} || \pi_\theta) \leq \delta$$

---

## PPO 核心思想

### 比率 (Ratio)

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

- $r_t(\theta) = 1$: 新旧策略相同
- $r_t(\theta) > 1$: 新策略更倾向于选择该动作
- $r_t(\theta) < 1$: 新策略更不倾向于选择该动作

### 裁剪目标

$$L^{CLIP}(\theta) = E\left[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)\right]$$

---

## 裁剪机制详解

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

### 直观理解

**当 A > 0 (好动作)**:
- ratio 被限制在 $[1-\epsilon, 1+\epsilon]$
- 防止过度增加好动作的概率

**当 A < 0 (坏动作)**:
- ratio 被限制在 $[1-\epsilon, 1+\epsilon]$
- 防止过度减少坏动作的概率

---

## PPO 完整目标函数

$$L(\theta) = E\left[L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s)\right]$$

其中：
- $L^{CLIP}(\theta)$: 裁剪的策略目标
- $L^{VF}(\theta)$: 价值函数损失
- $S[\pi_\theta](s)$: 熵正则化项
- $c_1, c_2$: 系数

---

## 代码实现

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

    return total_loss
```

---

## 超参数

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| clip_epsilon | 0.1-0.2 | 裁剪范围 |
| gamma | 0.99 | 折扣因子 |
| lam | 0.95 | GAE 参数 |
| learning_rate | 1e-4 - 3e-4 | 学习率 |
| n_epochs | 3-10 | 每次采集后的训练轮数 |
| batch_size | 64-256 | 小批量大小 |

---

## 关键要点

1. **PPO 是最流行的策略梯度算法之一**
2. **裁剪机制确保更新在安全范围内**
3. **GAE 平衡偏差和方差**
4. **广泛应用于游戏、机器人、LLM 等领域**
