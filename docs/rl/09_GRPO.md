# GRPO

> 原书 PDF 第 141–147 页
> 人话版：[Modelshit Class · GRPO：去掉评论家](https://modelshit-class.vercel.app/01-rl/09-grpo/)

## 1. 一句话

组内相对奖励替代 Critic，DeepSeek-R1 同款思路的简化版。

## 2. 章节结构

- 9.2 使用GRPO 玩倒立摆游戏
- 200 = 0.015。这就是我

## 3. 核心要点

1. 9. 组相对策略优化（GRPO） 9.1 GRPO 原理 GRPO Group Relative Policy Optimization：组相对策略优化 定理 9.1 — GRPO 的目标函数. 𝐽(𝜃)GRPO = 1 𝐺∑ 𝐺 𝑖=1 1 |𝜏𝑖| ∑ |𝜏𝑖| 𝑡=1 min[𝜌𝐴𝜏𝑖,𝑡, clip(𝑝, 1 −𝜀, 1 + 𝜀)𝐴𝜏𝑖,𝑡] 其中比值（重要性权重） 𝜌= 𝜋𝜃(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) 𝜋𝜃old(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) (9.1) 先来说明一下GRPO 目标函数中每个数学符号的含义： • 𝜋𝜃表示正在更新的策略。 • 𝜋𝜃old表示上一轮训练好的旧策略。 • 𝐺表示使用旧策略𝜋𝜃old采样的一组轨迹的数量，也就是如果我们使用旧策略采样了10 条轨迹，那 么𝐺= 10。 • 𝜏𝑖表示第𝑖条轨迹。 • |𝜏𝑖|表示第𝑖条轨迹的动作数量。 • 𝜋𝜃(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡)表示第𝑖条轨迹的第𝑡个时刻的状态为𝑠𝜏𝑖,𝑡，以及在这个状态下正在更新的策略𝜋𝜃采 取动作𝑎𝜏𝑖,𝑡的概率。 • 𝜋𝜃old(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡)表示第𝑖条轨迹的第𝑡个时刻的状态为𝑠...

2. 142 Chapter 9. 组相对策略优化（GRPO） 2 def __init__(self, action_size): 3 super().__init__() 4 self.l1 = nn.Linear(4, 128) 5 self.l2 = nn.Linear(128, action_size) 6 7 def forward(self, x): 8 x = F.relu(self.l1(x)) 9 x = F.softmax(self.l2(x), dim=1) 10 return x 智能体代码如下： 1 class Agent: Python 2 def __init__(self): 3 self.lr = 0.0002 4 self.action_size = 2 5 self.pi = PolicyNet(self.action_size) 6 self.optimizer = optim.Adam(self.pi.parameters(), lr=self.lr) 7 8 def get_action(self, state): 9 probs = se...

3. 9.2 使用GRPO 玩倒立摆游戏 143 25 26 return states, log_probs, actions, normalized_reward 这里的归一化奖励需要说一下，我们已经知道木杆坚持200 步不倒下，游戏就成功结束了。那么如 果木杆坚持了3 步就倒下，这条轨迹的奖励应该如何计算呢？这里我们选择3 200 = 0.015。这就是我 们给这条轨迹的奖励。 GRPO 的优势计算是和PPO 的优势计算有区别的地方。PPO 使用了价值函数网络评估每个动作的 价值，并且使用了广义优势估计（GAE）。而GRPO 创造性的提出了组相对优势。 也就是轨迹𝜏𝑖相对于组内其它轨迹的优势是多少？也就是如下 轨迹𝜏𝑖的归一化奖励是： 𝑅normalized 𝜏𝑖 = 𝐺(𝜏𝑖) 200 (9.2) 而一组轨迹的平均奖励是 rewardmean = 1 𝐺∑ 𝐺 𝑖=0 𝑅normalized 𝜏𝑖 (9.3) 那么轨迹𝜏𝑖相对于组内其它轨迹的优势为： 𝐴𝜏𝑖= 𝑅normalized 𝜏𝑖 −rewardmean rewardstd (9.4) 轨迹中每一个动作的优势就等于这个...

4. 144 Chapter 9. 组相对策略优化（GRPO） 9 std_reward = np.std(rewards) + 1e-8 10 # [轨迹0的组相对优势，轨迹1的组相对优势，...] 11 advantages = [(r - mean_reward) / std_reward for r in rewards] 12 13 return advantages 使用GRPO 算法更新策略的代码如下 1 class Agent: Python 2 ... 3 4 def update(self, trajectories): 5 advantages = self.calc_advantages_with_grpo(trajectories) 6 7 for step in range(20): 8 loss = 0.0 9 for traj, advantage in zip(trajectories, advantages): 10 """遍历组里面的每一条轨迹和对应的组内优势""" 11 states, log_probs, actions, _ = traj 1...

5. 9.2 使用GRPO 玩倒立摆游戏 145 13 agent.update(trajectories) 14 15 # 一组轨迹的平均奖励 16 avg_reward = sum(episode_rewards) / len(episode_rewards) 17 trial_num += 1 18 19 if avg_reward > 195: 20 print("训练结束，训练回合数：", trial_num) 21 return 22 else: 23 print(f"训练回合数：{trial_num}，平均奖励：{avg_reward}")

## 4. 与前后章关系

- 在线阅读：https://modelshit-class.vercel.app/01-rl/09-grpo/
- 本地对照：`python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 141 --end 147`
