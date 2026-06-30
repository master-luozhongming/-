# 强化学习基本概念

## 核心要素

### 智能体与环境

```
┌─────────────┐         ┌─────────────┐
│   智能体     │ ──────► │    环境      │
│   (Agent)    │ 动作a   │(Environment)│
│              │ ◄────── │             │
└─────────────┘ 状态s,r  └─────────────┘
```

---

## 状态、动作、奖励

| 概念 | 符号 | 含义 |
|------|------|------|
| 状态 | s, S_t | 环境的当前状态 |
| 动作 | a, A_t | 智能体采取的行为 |
| 奖励 | r, R_t | 即时反馈信号 |

---

## 策略 (Policy)

### 定义

策略是状态到动作的映射：

$$\pi(a|s) = P(a_t = a | s_t = s)$$

### 类型

- **确定性策略**: $a = \pi(s)$
- **随机策略**: $a \sim \pi(\cdot|s)$

---

## 轨迹 (Trajectory)

$$\tau = (S_0, A_0, R_0, S_1, A_1, R_1, S_2, A_2, R_2, ...)$$

---

## 回报 (Return)

$$G_t = R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + ... = R_t + \gamma G_{t+1}$$

其中 $\gamma \in [0, 1]$ 是折扣因子。

---

## 价值函数

### 状态价值函数

$$V_\pi(s) = E_\pi[G_t | S_t = s]$$

### 动作价值函数

$$Q_\pi(s, a) = E_\pi[G_t | S_t = s, A_t = a]$$

### 关系

$$V_\pi(s) = \sum_a \pi(a|s) Q_\pi(s, a)$$

---

## 贝尔曼方程

$$V_\pi(S_t) = E_\pi[R_t + \gamma V_\pi(S_{t+1}) | S_t]$$

---

## 探索与利用

```python
def epsilon_greedy(q_values, epsilon=0.1):
    if random.random() < epsilon:
        return random.randint(0, len(q_values) - 1)  # 探索
    else:
        return np.argmax(q_values)  # 利用
```

---

## 关键要点

1. **核心循环**: 状态 → 动作 → 奖励 → 新状态
2. **策略**: 决定在什么状态下采取什么动作
3. **价值函数**: 评估状态或动作的好坏
4. **贝尔曼方程**: 动态规划的基础
