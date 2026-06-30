# PPO 实现

## Actor-Critic 网络

```python
import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        features = self.shared(state)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value

    def get_action(self, state):
        action_probs, value = self.forward(torch.FloatTensor(state))
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate(self, states, actions):
        action_probs, values = self.forward(states)
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values.squeeze(), entropy
```

---

## PPO 算法

```python
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 lam=0.95, clip_epsilon=0.2, n_epochs=10, batch_size=64):
        self.policy = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def update(self, states, actions, old_log_probs, advantages, returns):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)

        total_loss = 0

        for _ in range(self.n_epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]

                new_log_probs, values, entropy = self.policy.evaluate(
                    states[idx], actions[idx]
                )

                ratio = torch.exp(new_log_probs - old_log_probs[idx])

                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                                    1 + self.clip_epsilon) * advantages[idx]

                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, returns[idx])
                entropy_loss = -entropy.mean()

                loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_loss += loss.item()

        return total_loss / self.n_epochs
```

---

## 训练循环

```python
def train_ppo(env, ppo, num_iterations=1000, max_steps=2048):
    episode_rewards = []

    for iteration in range(num_iterations):
        # 收集轨迹
        states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

        state = env.reset()
        for _ in range(max_steps):
            action, log_prob, value = ppo.policy.get_action(state)
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state
            if done:
                state = env.reset()

        # 计算 GAE
        advantages, returns = ppo.compute_gae(rewards, values, values[1:] + [0], dones)

        # 更新
        loss = ppo.update(states, actions, log_probs, advantages, returns)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}, Loss: {loss:.4f}")

    return episode_rewards
```

---

## 关键要点

1. **Actor-Critic** 共享特征层
2. **GAE** 计算优势估计
3. **裁剪目标** 确保更新稳定
4. **多轮更新** 提高样本效率
