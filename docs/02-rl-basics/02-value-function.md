# 价值函数

## 状态价值函数 V(s)

### 定义

在状态 $s$ 下，按照策略 $\pi$ 行动，未来预期回报：

$$V_\pi(s) = E_\pi[G_t | S_t = s] = E_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k} | S_t = s\right]$$

### 直观理解

- $V(s)$ 衡量状态 $s$ 有多"好"
- 高 $V(s)$ 意味着从该状态出发能获得高回报

---

## 动作价值函数 Q(s,a)

### 定义

在状态 $s$ 下采取动作 $a$，然后按照策略 $\pi$ 行动，未来预期回报：

$$Q_\pi(s, a) = E_\pi[G_t | S_t = s, A_t = a]$$

### 与 V(s) 的关系

$$V_\pi(s) = \sum_a \pi(a|s) Q_\pi(s, a)$$

---

## 优势函数 A(s,a)

### 定义

$$A_\pi(s, a) = Q_\pi(s, a) - V_\pi(s)$$

### 直观理解

- $A(s, a) > 0$: 动作 $a$ 比平均水平好
- $A(s, a) < 0$: 动作 $a$ 比平均水平差
- $A(s, a) = 0$: 动作 $a$ 是平均水平

---

## 代码实现

```python
import numpy as np

def compute_value_function(env, policy, gamma=0.99, theta=1e-6):
    """
    计算状态价值函数（迭代策略评估）

    Args:
        env: 环境
        policy: 策略
        gamma: 折扣因子
        theta: 收敛阈值

    Returns:
        V: 状态价值函数
    """
    n_states = env.observation_space.n
    V = np.zeros(n_states)

    while True:
        delta = 0
        for s in range(n_states):
            v = V[s]

            # 计算新的价值
            new_v = 0
            for a in range(env.action_space.n):
                action_prob = policy[s][a]
                for prob, next_state, reward, done in env.P[s][a]:
                    if done:
                        new_v += action_prob * prob * reward
                    else:
                        new_v += action_prob * prob * (reward + gamma * V[next_state])

            V[s] = new_v
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V
```

---

## 关键要点

1. **V(s)** 评估状态的好坏
2. **Q(s,a)** 评估状态-动作对的好坏
3. **A(s,a)** 衡量动作相对于平均水平的优势
4. 价值函数是强化学习的核心概念
