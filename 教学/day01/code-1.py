import gymnasium as gym  # 导入 Gymnasium 库（Gym 的维护版本）

print(gym.__version__)  # 打印 Gymnasium 版本号
# 强化学习之父：Richard Sutton
env = gym.make("CartPole-v1")  # 创建 CartPole-v1 环境
state, _ = env.reset()  # 重置为 S_0 状态（Gymnasium 新 API 返回元组）
print("S_0: ", state)  # 打印初始状态

action_space = env.action_space  # 获取动作空间（两个动作：0=向左，1=向右）
print(action_space)  # 打印动作空间信息

# 选择动作：向左推
action = 0
# 采取向左推的动作
next_state, reward, terminated, truncated, info = env.step(action)  # 执行动作（Gymnasium 新 API）
done = terminated or truncated  # 判断是否结束
print("S_1: ", next_state)  # 打印下一个状态
print("R_0: ", reward)  # 打印获得的奖励
print("是否结束：", done)  # 打印是否结束
