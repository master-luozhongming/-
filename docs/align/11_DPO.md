# DPO

> 原书 PDF 第 155–180 页
> 人话版：[Modelshit Class · DPO：不用强化学习也能对齐](https://modelshit-class.vercel.app/02-align/11-dpo/)

## 1. 一句话

把偏好学习改写成分类损失，省掉奖励模型和 PPO 采样循环。

## 2. 章节结构

- 11.1.1 偏好数据集
- 11.1.2 DPO 目标函数
- 11.1 DPO 理论介绍以及DPO 存在的问题
- 11.2.1 第一步：对预训练模型进行基于指令的监督微调（SFT）
- 11.2 DPO 实战
- 11.2.2 第二步：使用DPO 算法对SFT 后的模型进行微调

## 3. 核心要点

1. 11. 使用DPO 微调大语言模型 11.1 DPO 理论介绍以及DPO 存在的问题 提示 • DPO：Direct Preference Optimization • 直接偏好优化：你的大语言模型实际上是一个奖励模型 11.1.1 偏好数据集 DPO 需要偏好数据集来微调LLM。偏好数据集的格式如下： 1 { JSON 2 "prompt": "这部电影怎么样？", 3 "chosen": "这部电影很好看。", 4 "rejected": "这部电影不好看。" 5 } 数据集构建的方法： • 针对同一个prompt，通过调整温度，让LLM 输出不同的回答。然后让数据标注工程师来标注对 不同回答的偏好。温度示例网站。 • 在使用ChatGPT 时，你可能注意到，偶尔会被要求在两个相似的答案中选择一个来继续对话。这 个偏好会被记录下来，并用于在未来的偏好调整中改进模型。同样。 • 手工标注 • 合成数据：通过写提示词来让LLM 生成偏好数据集（需要人工审核）。 • 使用网上的开源数据集 • … 数据集容易存在的问题： 1. 正负例区分不明显 1 { JSON 2 "prompt"...

2. 156 Chapter 11. 使用DPO 微调大语言模型 3 "chosen": "这部电影很好看。", 4 "rejected": "这部电影挺好看。" 5 } 人类都无法识别哪个应该是正例，哪个应该是负例。 1. 数据集中存在偏好循环 1 { JSON 2 "prompt": "这部电影怎么样？", 3 "chosen": "这部电影很好看。", 4 "rejected": "这部电影很差。" 5 }, 6 { 7 "prompt": "这部电影怎么样？", 8 "chosen": "这部电影很差。", 9 "rejected": "这部电影很好看。" 10 } 当模型看到以上数据时，就不知道人类的偏好是什么了。无法学到任何东西。因为第一个人标注的 数据的偏好是𝐴≻𝐵，第二个人标注的数据的偏好是𝐵≻𝐴，那么模型看到的偏好是𝐴≻𝐵≻𝐴。这 就是偏好循环。 11.1.2 DPO 目标函数 𝐽DPO(𝜋𝜃; 𝜋ref) = 𝔼(𝑦𝑤,𝑦𝑙,𝑥)∼𝒟︀[log(𝜎(𝛽(log 𝜋𝜃(𝑦𝑤|𝑥) 𝜋𝜃(𝑦𝑙|𝑥) −log 𝜋ref(𝑦𝑤|𝑥) 𝜋ref(𝑦𝑙|𝑥) )))] 正在训...

3. 11.1 DPO 理论介绍以及DPO 存在的问题 157 通过观察目标函数，我们知道在最开始训练时：𝜋𝜃= 𝜋ref，所以log 𝜋𝜃(𝑦𝑤|𝑥) 𝜋ref(𝑦𝑤|𝑥) −log 𝜋𝜃(𝑦𝑙|𝑥) 𝜋ref(𝑦𝑙|𝑥) = 0。但由 于外层有𝜎函数，所以不影响反向传播更新网络（𝜎(0) = 0.5）。 通过观察目标函数，我们可以知道随着训练的进行，𝜋𝜃(𝑦𝑤|𝑥) 𝜋𝜃(𝑦𝑙|𝑥) 会越来越大。因为𝜋ref(𝑦𝑤|𝑥) 𝜋ref(𝑦𝑙|𝑥) 是一个常 数（作为正则化项存在）。 DPO 存在的问题 1. 𝜋𝜃(𝑦𝑤|𝑥) 𝜋𝜃(𝑦𝑙|𝑥) 的分子和分母可能同时增大或者减小。例如分母增大了1.5 倍，分子增大了3 倍。 那么就导致了LLM 输出人类偏好的回答的概率和输出人类讨厌的回答的概率都增加了。 2. 𝜋𝜃(𝑦𝑤|𝑥) 𝜋𝜃(𝑦𝑙|𝑥) 随着训练，分子越来越大，分母越来越小，结果导致了𝜋𝜃(𝑦𝑤|𝑥)趋近于1，𝜋𝜃(𝑦𝑙|𝑥) 趋近于0，正则化项𝜋ref(𝑦𝑤|𝑥) 𝜋ref(𝑦𝑙|𝑥) 没有起到作用，最终LLM 彻底失去了探索能力。也就是出现了 “过拟合”的问题。DPO 容易过...

4. 158 Chapter 11. 使用DPO 微调大语言模型 . . . . . . . . . . . . Question Answer 目标：在给定问题（Question）的前提下，使得模型产生该回答（Answer）的概率最大化，也就是 指令微调的优化目标 等价于 等价于 等价于 最终优化目标Loss Sequence 图 11.1 SFT 的损失函数 我们要注意的是在基于指令的监督微调中，我们只针对答案部分计算损失。

5. 11.2 DPO 实战 159 . . . . . . . . . . . . Question Answer 损失函数的实现方法 Sequence 模型推理：本质上就是预测下一个token 获取 x(Question) 中每个 token 的 Logits 以及 y(具体是Answer[:-1]) 针对每个 token 计算对应的 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . -log 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ⨂ Answer Mask Sum()/Answer_Mask.sum() 也就是 Loss 后面的交给梯度下降算法（AdamW）进行参数优化 图 11.2 基于指令的监督微调只针对答案部分计算损失 我们将损失的计算推广到批次（batch）。 . . . . . . . . . . . . Question Answer ...

6. 160 Chapter 11. 使用DPO 微调大语言模型 7 import time 接下来我们导入模型和分词器。 1 device = "cuda" Python 2 model_path = "./Qwen3-0.6B-Base" 3 4 model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto", device_map="auto") 5 tokenizer = AutoTokenizer.from_pretrained(model_path) 然后我们设置一下生成文本的参数。保证测试的一致性。 其中151645 为<|im_end|> ，151643 为<|endoftext|> 。 1 model.generation_config.do_sample = True Python 2 model.generation_config.eos_token_id = [151645, 151643] 3 model.generation_config.pad_token_id = 151643 ...

7. 11.2 DPO 实战 161 16 train_data = [] 17 i = 0 18 while True: 19 data = ultrachat_200k_data["train_sft"][i]["messages"] 20 # 添加系统提示词 21 data.insert( 22 0, 23 {"content": "You are a helpful assistant", "role": "system"} 24 ) 25 input_ids = tokenize_and_format(data) 26 train_data.append(input_ids) 27 i += 1 28 if i % 1000 == 0: 29 print(f"已处理{i}条数据") 30 if i == 50000: 31 break 接下来我们编写一下学习率的线性预热和余弦衰减的函数。 1 def linear_warmup(current_step, warmup_steps, max_lr): Python 2 if current_step < warmup_step...

8. 162 Chapter 11. 使用DPO 微调大语言模型 7 "content": "我是左元。", 8 "role": "assistant" 9 }, 10 { 11 "content": "你会强化学习吗？", 12 "role": "user" 13 }, 14 { 15 "content": "略知一二。", 16 "role": "assistant" 17 } 18 ] 经过模型的对话模板格式化之后是 1 <|im_start|>system 2 You are a helpful assistant<|im_end|> 3 <|im_start|>user 4 你是谁？<|im_end|> 5 <|im_start|>assistant 6 我是左元。<|im_end|> 7 <|im_start|>user 8 你会强化学习吗？<|im_end|> 9 <|im_start|>assistant 10 <think></think>略知一二。<|im_end|> 代码如下： 1 def create_answer_mask(input_ids, tokeniz...

## 4. 与前后章关系

- 在线阅读：https://modelshit-class.vercel.app/02-align/11-dpo/
- 本地对照：`python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 155 --end 180`
