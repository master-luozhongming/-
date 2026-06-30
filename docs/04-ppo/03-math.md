# PPO 数学推导

## KL 散度

### 定义

$$KL(P || Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

### 性质

- $KL(P || Q) \geq 0$
- $KL(P || Q) = 0$ 当且仅当 $P = Q$
- $KL(P || Q) \neq KL(Q || P)$

---

## 重要性采样

### 基本思想

用分布 $q$ 的样本来估计分布 $p$ 的期望：

$$E_{x \sim p}[f(x)] = E_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right]$$

### 在策略梯度中的应用

$$E_{\tau \sim \pi_\theta}[R(\tau)] = E_{\tau \sim \pi_{\theta_{old}}}\left[\frac{P(\tau; \theta)}{P(\tau; \theta_{old})} R(\tau)\right]$$

---

## 替代目标函数

### CPI 目标

$$L^{CPI}(\theta) = E_{\tau \sim \pi_{\theta_{old}}}\left[\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} A_t\right]$$

### 问题

无约束的优化可能导致策略变化太大。

---

## PPO 裁剪目标

### 公式

$$L^{CLIP}(\theta) = E\left[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)\right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$

### 推导

当 $A_t > 0$（好动作）：
- 我们想增加 $r_t(\theta)$
- 但裁剪限制了 $r_t(\theta) \leq 1 + \epsilon$

当 $A_t < 0$（坏动作）：
- 我们想减少 $r_t(\theta)$
- 但裁剪限制了 $r_t(\theta) \geq 1 - \epsilon$

---

## 全变差距离

### 定义

$$D_{TV}(P || Q) = \frac{1}{2} \sum_x |P(x) - Q(x)|$$

### 与 KL 散度的关系

$$D_{TV}(P || Q)^2 \leq KL(P || Q) \leq 2 D_{TV}(P || Q)^2$$

---

## 误差上界

### 策略更新的误差

$$|J(\theta) - J(\theta_{old})| \leq C \cdot D_{TV}(\pi_\theta || \pi_{\theta_{old}})$$

### PPO 的保证

裁剪机制确保 $D_{TV}$ 在可控范围内，从而保证更新是"安全"的。

---

## 关键要点

1. **重要性采样**允许重用旧数据
2. **KL 散度**衡量策略差异
3. **裁剪机制**限制策略变化幅度
4. **理论保证**PPO 的更新是稳定的
