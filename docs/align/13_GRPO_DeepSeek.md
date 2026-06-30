# 用 GRPO 复刻 DeepSeek-R1

> 原书 PDF 第 203–239 页
> 人话版：[Modelshit Class · 用 GRPO 复刻 DeepSeek-R1](https://modelshit-class.vercel.app/02-align/13-grpo-deepseek/)

## 1. 一句话

DAPO/GRPO 微调 Qwen，奖励函数设计与推理链场景。

## 2. 章节结构

- 13.2.1 实现思路及任务要求
- 13.2 使用DAPO（GRPO 的变种）微调Qwen2.5-3B-Instruct
- 44 减去 10 等于 34 ，最后将这两步结果利用小括号组合在一起形成数学表达式为 (81 - 37) - 10 等于 34 。
- 13.2.2 奖励函数的设计
- 0 的奖励。
- 13.2.3 代码实现
- 13.3.1 医疗思维链
- 13.3 GRPO 应用场景
- 13.3.2 Text-To-SQL

## 3. 核心要点

1. 13. 使用GRPO 微调大语言模型——复刻 DeepSeek-R1 强化学习已成为增强大语言模型初始训练效果的强大工具，尤其是在推理密集型任务中。 DeepSeek 最近在 DeepSeek-Math 和 DeepSeek-R1 模型上取得的突破，展现了 RL 在提 升 LLM 数学推理和问题解决能力方面的巨大潜力。 这些成就得益于一种名为“组相对策略优化”（GRPO）的创新强化学习方法，该方法解决了将强 化学习应用于语言模型的独特挑战。我们将深入探讨 GRPO 的工作原理，以及它为何代表了 LLM 训练的重大进步。 13.1 GRPO 简介 GRPO 目标函数如下： 𝐽(𝜃)GRPO = 1 𝐺∑ 𝐺 𝑖=1 1 |𝜏𝑖| ∑ |𝜏𝑖| 𝑡=1 min[𝑝𝐴𝜏𝑖,𝑡, clip(𝑝, 1 −𝜀, 1 + 𝜀)𝐴𝜏𝑖,𝑡] −𝛽𝐷KL[𝜋𝜃∥𝜋ref] 其中比值 𝑝= 𝜋𝜃(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) 𝜋𝜃old(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) (13.1) 先来说明一下GRPO 目标函数中每个数学符号的含义： • 𝜋𝜃表示正在更新的策略。 • 𝜋𝜃old表示上一轮训练好的旧策略。 •...

2. 204 Chapter 13. 使用GRPO 微调大语言模型——复刻DeepSeek-R1 13.2 使用DAPO（GRPO 的变种）微调Qwen2.5-3B-Instruct 13.2.1 实现思路及任务要求 我们要从零复刻一个类似DeepSeek-R1 的模型，产生思维链！ 我们的实现不采用原始的GRPO 算法，而是采用字节提出的 DAPO 算法，DAPO 对原始的 GRPO 有如下几项改进： 1. token 级的策略梯度损失：每个token 在策略梯度损失中具有同等权重。也就是每个 token 的优势等于token 所在轨迹的组相对优势！ 2. 移除KL 散度：策略梯度损失中不再使用KL 散度。由于我们不再需要参考策略网络𝜋ref，这 可以减少 GPU 内存的使用。 我们的算法伪代码如下： DAPO 算法伪代码 1. 对于每个训练步骤，随机选取𝑁个问题：𝑞1, 𝑞2, …, 𝑞𝑁。 2. 对于每个问题𝑞𝑖，采样𝐺条回答（轨迹）：𝑎𝑖,1, 𝑎𝑖,2, …, 𝑎𝑖,𝐺 。 |> 𝐺为一组轨迹中轨迹的 数量 3. 计算每个回答𝑎𝑖,𝑗的奖励𝑟𝑖,𝑗。 • 𝑎𝑖,𝑗为第i 个问...

3. 13.2 使用DAPO（GRPO 的变种）微调Qwen2.5-3B-Instruct 205 伪代码中第6 步的解释 GRPO 目标函数最内层为 min [ 𝜋𝜃(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) 𝜋𝜃old(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) 𝐴𝜏𝑖,𝑡, clip ( 𝜋𝜃(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) 𝜋𝜃old(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) , 1 −𝜀, 1 + 𝜀 ) 𝐴𝜏𝑖,𝑡 ] (13.5) 如果GRPO 使用旧策略采集的轨迹只更新一次策略的话，相当于在𝜃= 𝜃old处进行求导（梯度）。 由于此时𝜃= 𝜃old，所以裁剪不会发生。也就是如下： ∇𝜃min [ 𝜋𝜃(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) 𝜋𝜃old(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) 𝐴𝜏𝑖,𝑡, clip ( 𝜋𝜃(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) 𝜋𝜃old(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) , 1 −𝜀, 1 + 𝜀 ) 𝐴𝜏𝑖,𝑡 ] |𝜃=𝜃old = ∇𝜃 { 𝜋𝜃(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) 𝜋𝜃old(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡) 𝐴𝜏𝑖,𝑡 } |𝜃=𝜃old = ∇𝜃𝜋𝜃(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡)|𝜃=𝜃old 𝜋𝜃old(𝑎𝜏𝑖,𝑡|𝑠𝜏𝑖,𝑡...

4. 206 Chapter 13. 使用GRPO 微调大语言模型——复刻DeepSeek-R1 使用这些数字 [37 81 10]，创建一个等于 34 的等式。你可以使用基本算术运算（+、-、*、/），每个数字只能使 用一次。在 <think> </think> 标签中展示你的解题过程。并在 <answer> </answer> 标签中返回最终答案， 例如 <answer> (1 + 2) / 3 </answer>。<|im_end|> <|im_start|>assistant 让我一步步来解决这个问题。 <think>可以通过组合给定的数字 37 81 10 使它们满足运算为 34 。首先通过 81 减去 37 得到 44 ，再进行 减法运算 44 减去 10 等于 34 ，最后将这两步结果利用小括号组合在一起形成数学表达式为 (81 - 37) - 10 等于 34 。 </think> <answer> (81 - 37) - 10 </answer><|im_end|> 13.2.2 奖励函数的设计 在使用GRPO 玩倒立摆游戏时，我们的奖励很简单，例如一条轨迹执行了5 ...

5. 13.2 使用DAPO（GRPO 的变种）微调Qwen2.5-3B-Instruct 207 3 from pathlib import Path 4 from typing import Optional, Tuple, Union 5 6 import torch 7 import torch.nn.functional as F 8 from torch import nn 9 10 11 @dataclass 12 class Qwen2Config: 13 attention_dropout: float = 0.0 14 bos_token_id: int = 151643 15 eos_token_id: int = 151645 16 hidden_act: str = "silu" 17 hidden_size: int = 2048 18 initializer_range: float = 0.02 19 intermediate_size: int = 11008 20 max_position_embeddings: int = 32768 21 max...

6. 208 Chapter 13. 使用GRPO 微调大语言模型——复刻DeepSeek-R1 53 def rotate_half(x): 54 """Rotates half the hidden dims of the input.""" 55 x1 = x[..., : x.shape[-1] // 2] 56 x2 = x[..., x.shape[-1] // 2 :] 57 return torch.cat((-x2, x1), dim=-1) 58 59 60 def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2): 61 cos = cos.unsqueeze(unsqueeze_dim) 62 sin = sin.unsqueeze(unsqueeze_dim) 63 q_embed = (q * cos) + (rotate_half(q) * sin) 64 k_embed = (k * cos) + (rotate_half(k) * sin) 65 return q_embed, k_embed ...

7. 13.2 使用DAPO（GRPO 的变种）微调Qwen2.5-3B-Instruct 209 103 def init_kv_cache( 104 self, 105 max_batch_size: int, 106 max_seq_len: int, 107 dtype: torch.dtype, 108 device: torch.device, 109 ): 110 cache_shape = (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim) 111 cache_k = torch.zeros(cache_shape, dtype=dtype, device=device) 112 cache_v = torch.zeros(cache_shape, dtype=dtype, device=device) 113 self.register_buffer("cache_k", cache_k, persistent=False) 114 self.register_buffer("cache_v", ...

8. 210 Chapter 13. 使用GRPO 微调大语言模型——复刻DeepSeek-R1 153 enable_gqa=True, 154 ).transpose(1, 2) 155 output = output.reshape(bsz, seqlen, -1) 156 return self.o_proj(output) 157 158 159 class FeedForward(nn.Module): 160 def __init__( 161 self, 162 dim: int, 163 intermediate_size: int, 164 ): 165 super().__init__() 166 self.up_proj = nn.Linear(dim, intermediate_size, bias=False) 167 self.down_proj = nn.Linear(intermediate_size, dim, bias=False) 168 self.gate_proj = nn.Linear(dim, intermediate_size, bia...

## 4. 与前后章关系

- 在线阅读：https://modelshit-class.vercel.app/02-align/13-grpo-deepseek/
- 本地对照：`python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 203 --end 239`
