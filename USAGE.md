# 使用说明

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 或者
pip install gym torch numpy matplotlib
```

### 2. 运行代码示例

#### 倒立摆环境基础
```bash
python code/rl_basics/cartpole_env.py
```

#### REINFORCE 算法
```bash
python code/policy_gradient/reinforce.py
```

#### PPO 算法
```bash
python code/ppo/ppo.py
```

### 3. 阅读文档

- [强化学习基本概念](02_RL_Basics/concepts.md)
- [REINFORCE 算法](03_Policy_Gradient/reinforce.md)
- [PPO 理论](04_PPO/theory.md)
- [PPO 实现](04_PPO/implementation.md)
- [GRPO 算法](05_RLHF/grpo.md)

## 项目结构

```
reinforcement/
├── README.md                    # 项目总览
├── USAGE.md                     # 使用说明（本文件）
├── LICENSE                      # MIT 许可证
├── .gitignore                   # Git 忽略文件
├── pyproject.toml               # 项目配置
│
├── 01_LLM/                      # 大语言模型
│   └── microgpt.md
│
├── 02_RL_Basics/                # 强化学习基础
│   └── concepts.md
│
├── 03_Policy_Gradient/          # 策略梯度法
│   └── reinforce.md
│
├── 04_PPO/                      # 近端策略优化
│   ├── theory.md
│   └── implementation.md
│
├── 05_RLHF/                     # RLHF
│   └── grpo.md
│
└── code/                        # 代码示例
    ├── __init__.py
    ├── rl_basics/
    │   └── cartpole_env.py
    ├── policy_gradient/
    │   └── reinforce.py
    └── ppo/
        └── ppo.py
```

## 学习路径

1. **入门**: 阅读 [强化学习基本概念](02_RL_Basics/concepts.md)
2. **实践**: 运行 [倒立摆环境示例](code/rl_basics/cartpole_env.py)
3. **基础算法**: 学习 [REINFORCE](03_Policy_Gradient/reinforce.md)
4. **进阶算法**: 学习 [PPO](04_PPO/theory.md)
5. **应用**: 了解 [GRPO](05_RLHF/grpo.md) 在 LLM 中的应用

## 常见问题

### Q: 如何在自己的环境中使用这些算法？

A: 参考 `code/` 目录下的代码示例，将环境接口适配到你的环境即可。

### Q: 如何调整超参数？

A: 每个算法文件都有详细的超参数说明，可以根据具体任务进行调整。

### Q: 如何贡献代码？

A: Fork 本仓库，创建新分支，提交 PR 即可。

## 参考资料

- [OpenAI Gym](https://gym.openai.com)
- [PyTorch](https://pytorch.org)
- [Spinning Up in Deep RL](https://spinningup.openai.com)
- 原始文档: 《大语言模型、强化学习和多模态教程》- 左元

## 联系方式

如有问题，欢迎提 Issue 或联系作者。
