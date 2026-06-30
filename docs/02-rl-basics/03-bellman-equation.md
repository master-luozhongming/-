# 贝尔曼方程

## 贝尔曼期望方程

### 状态价值函数

$$V_\pi(s) = E_\pi[R_t + \gamma V_\pi(S_{t+1}) | S_t = s]$$

### 动作价值函数

$$Q_\pi(s, a) = E_\pi[R_t + \gamma Q_\pi(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]$$

---

## 直观理解

### 状态价值

$$\text{当前状态的价值} = \text{即时奖励} + \text{折扣因子} \times \text{下一状态的价值}$$

### 例子

```
你对公司的评估 = 现在公司给你的薪资 + 折扣因子 × 你对公司未来的评估
```

---

## 贝尔曼最优方程

### 最优状态价值

$$V^*(s) = \max_a E[R_t + \gamma V^*(S_{t+1}) | S_t = s, A_t = a]$$

### 最优动作价值

$$Q^*(s, a) = E[R_t + \gamma \max_{a'} Q^*(S_{t+1}, a') | S_t = s, A_t = a]$$

---

## 推导

### 从回报到贝尔曼方程

$$V_\pi(s) = E_\pi[G_t | S_t = s]$$

$$= E_\pi[R_t + \gamma G_{t+1} | S_t = s]$$

$$= E_\pi[R_t + \gamma V_\pi(S_{t+1}) | S_t = s]$$

---

## 动态规划

### 策略评估

```python
def policy_evaluation(env, policy, gamma=0.99, theta=1e-6):
    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]

            # 贝尔曼期望方程
            new_v = 0
            for a, action_prob in enumerate(policy[s]):
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

### 策略改进

```python
def policy_improvement(env, V, gamma=0.99):
    policy = np.zeros([env.observation_space.n, env.action_space.n])

    for s in range(env.observation_space.n):
        q_values = np.zeros(env.action_space.n)

        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[s][a]:
                if done:
                    q_values[a] += prob * reward
                else:
                    q_values[a] += prob * (reward + gamma * V[next_state])

        # 贪心策略
        best_action = np.argmax(q_values)
        policy[s] = np.eye(env.action_space.n)[best_action]

    return policy
```

---

## 关键要点

1. **贝尔曼方程**是强化学习的数学基础
2. **递推关系**：当前价值依赖于下一状态的价值
3. **动态规划**：利用贝尔曼方程求解价值函数
4. **策略迭代**：评估 + 改进交替进行
