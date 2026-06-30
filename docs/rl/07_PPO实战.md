# PPO 实战

> 原书 PDF 第 119–124 页
> 人话版：[Modelshit Class · PPO 实战](https://modelshit-class.vercel.app/01-rl/07-ppo-practice/)

## 1. 一句话

Actor-Critic 网络、GAE 计算、PPO 训练循环在 CartPole 上的完整实现。

## 2. 章节结构

- 7.2 PPO 算法的实现
- 7.4 训练循环

## 3. 核心要点

1. 7. PPO 实战 7.1 网络结构的定义 • 演员网络：策略网络 • 评论家网络：价值函数网络 策略网络和价值函数网络的定义 Python 1 class PolicyNet(nn.Module): 2 def __init__(self, action_size): 3 super().__init__() 4 self.l1 = nn.Linear(4, 128) 5 self.l2 = nn.Linear(128, action_size) 6 7 def forward(self, x): 8 x = F.relu(self.l1(x)) 9 x = F.softmax(self.l2(x), dim=1) 10 return x 11 12 class ValueNet(nn.Module): 13 def __init__(self): 14 super().__init__() 15 self.l1 = nn.Linear(4, 128) 16 self.l2 = nn.Linear(128, 1) 17 18 def forward(self, x): 19 x...

2. 120 Chapter 7. PPO 实战 2 def __init__(self): 3 self.gamma = 0.98 4 self.lr_pi = 0.001 5 self.lr_v = 0.02 6 self.action_size = 2 7 8 self.pi = PolicyNet(self.action_size) 9 self.v = ValueNet() 10 11 self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi) 12 self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v) 13 14 def get_action(self, state): 15 probs = self.pi(torch.tensor(state).unsqueeze(0)).squeeze(0) 16 m = Categorical(probs) 17 action = m.sample().item() 18 return...

3. 7.2 PPO 算法的实现 121 有了旧策略采样的轨迹的所有详细信息，我们就可以来实现PPO 算法了。代码如下。 1 class Agent: Python 2 ... 3 4 def update(self, trajectory): 5 states, next_states, actions, action_probs, rewards, dones = trajectory 6 # [𝑠0, 𝑠1, …, 𝑠𝑇−1] 7 states = torch.tensor(states) 8 # [𝑎0, 𝑎1, …, 𝑎𝑇−1] 9 actions = torch.tensor(actions).view(-1, 1) 10 # [𝑅0, 𝑅1, …, 𝑅𝑇−1] 11 rewards = torch.tensor(rewards).view(-1, 1) 12 # [𝑠1, 𝑠2, …, 𝑠𝑇] 13 next_states = torch.tensor(next_states) 14 # [False1, False2, …, True𝑇] 15 dones = torc...

4. 122 Chapter 7. PPO 实战 49 self.optimizer_v.zero_grad() 50 pi_loss.backward() 51 v_loss.backward() 52 self.optimizer_pi.step() 53 self.optimizer_v.step() 比值的计算方法 ∵𝐴= 𝑒log 𝐴 ∴𝐴 𝐵= 𝑒log 𝐴−log 𝐵= 𝑒log 𝐴 𝑒log 𝐵 (7.3) 7.3 广义优势估计的计算 1 def compute_gae(gamma, td_delta): Python 2 # 𝛿𝑡 3 td_delta = td_delta.detach().numpy() 4 gae_list = [] 5 last_gae = 0.0 6 lmbda = 0.95 7 # 𝐴𝑡= 𝛿𝑡+ 𝛾𝜆𝐴𝑡+1 8 for delta in td_delta[::-1]: 9 last_gae = gamma * lmbda * last_gae + delta 10 gae_list.append(last_gae) 11 gae_lis...

5. 7.4 训练循环 123 21 agent = Agent() 22 return_list, episode_list = train(env, agent) 23 plot_loss(episode_list, return_list, "ppo-loss.pdf") 24 test_agent(agent, env) 25 26 if __name__ == "__main__": 27 main() 获得的奖励可视化如下： 图 7.1 PPO 算法获得的奖励

## 4. 与前后章关系

- 在线阅读：https://modelshit-class.vercel.app/01-rl/07-ppo-practice/
- 本地对照：`python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 119 --end 124`
