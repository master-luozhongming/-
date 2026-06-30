# 用 PPO 复刻 InstructGPT

> 原书 PDF 第 181–202 页
> 人话版：[Modelshit Class · 用 PPO 复刻 InstructGPT](https://modelshit-class.vercel.app/02-align/12-ppo-instructgpt/)

## 1. 一句话

SFT → 奖励模型 → PPO 三阶段，RLHF 工业标准管线。

## 2. 章节结构

- 12.2 即时奖励𝑅𝑡的计算
- 12.3.1 监督微调（SFT）
- 12.3 PPO 微调LLM 实战
- 12.3.2 训练奖励模型（Reward Model）
- 20 and len(x["review"]) < 1024)
- 12.3.3 对gpt2-sft 进行PPO 微调

## 3. 核心要点

1. 12. 使用PPO 微调大语言模型--复刻 InstructGPT 12.1 InstructGPT 训练流程 • Step-1：SFT，Supervised Fine-Tuning，有监督微调。顾名思义，它是在有监督（有标 注）数据上微调训练得到的。这里的监督数据其实就是输入Prompt，输出相应的回复，只不过 这里的回复是人工编写的。这个工作要求比一般标注要高，其实算是一种创作了。 • Step-2：RM，Reward Model，奖励模型。具体来说，一个Prompt 丢给前一步的SFT，输出 若干个（4-9 个）回复，由标注人员对这些回复进行排序。然后从4-9 个中每次取2 个，因为 是有序的，就可以用来训练这个奖励模型，让模型学习到这个好坏评价。这一步非常关键，它就 是所谓的Human Feedback，引导下一步模型的进化方向。 • Step-3：RL，Reinforcement Learning，强化学习，使用PPO 策略进行训练。PPO， Proximal Policy Optimization，近端策略优化，是一种强化学习优化方法，它背后的主 要思想是避免每次太大...

2. 182 Chapter 12. 使用PPO 微调大语言模型--复刻InstructGPT 图 12.1 InstructGPT 训练流程 12.2 即时奖励𝑅𝑡的计算 𝐽PPO(𝜃) = 𝔼[min( 𝜋𝜃(𝑎|𝑠) 𝜋𝜃old(𝑎|𝑠)𝐴, clip( 𝜋𝜃(𝑎|𝑠) 𝜋𝜃old(𝑎|𝑠), 1 −𝜀, 1 + 𝜀)𝐴)] (12.1) PPO 的目标函数我们已经很熟悉了，优势𝐴如果使用1 步TD 误差的话是：𝛿= 𝑅𝑡+ 𝛾𝑉(𝑆𝑡+1) − 𝑉(𝑆𝑡)。 在倒立摆环境中，只要木杆不倒下，那么𝑅𝑡= 1。但是在大语言模型这个环境中，情况就要复杂 多了。在大语言模型中，策略模型LLM 采取的动作是输出一个token，那么输出一个token，我 们应该给什么奖励𝑅𝑡呢？

3. 12.2 即时奖励𝑅𝑡的计算 183 This movie is very good and worth <eos> Prompt very good and worth <eos> 图 12.2 输出一个token（采取动作），针对输出的这个token，怎么给即时奖励𝑅𝑡？ 在RLHF 中，奖励模型（Reward Model）的作用是针对一条完整的补全给出分数的。 This movie is very good and worth watching. 奖励模型（Reward Model） 0.9分！ 图 12.3 奖励模型是针对一条完整的补全给出得分的 在 InstructGPT (Ouyang et al., 2022) 的 PPO 阶段，奖励函数的设计非常精妙。它不 仅仅是奖励模型（Reward Model, RM）给出的分数，还包含了一个至关重要的惩罚项。 即时奖励的计算公式分两种情况： • <eos_token> 之前的token 的奖励如下计算： 𝑅𝑡= −𝛽log 𝜋𝜃(𝑦𝑡|𝑥, 𝑦<𝑡) 𝜋ref(𝑦𝑡|𝑥, 𝑦<𝑡) (12.2) • 最后一个token 也就...

4. 184 Chapter 12. 使用PPO 微调大语言模型--复刻InstructGPT 在标准的InstructGPT 实现中，奖励模型（RM）的分数通常只加在最后一个 token 上，而 KL 惩罚是加在每一个 token 上的。 让我们把这个过程像切蛋糕一样切开来看： 1. 奖励的时间步分配 假设模型生成了一个长度为 𝑇 的句子：𝑦= [𝑦1, 𝑦2, …, 𝑦𝑇]。 在 PPO 的每一个时间步 𝑡，智能体获得的即时奖励 𝑅𝑡 是这样计算的： 时间步 (token) 即时奖励 𝑅𝑡 的构成 解释 中间token(𝑡< 𝑇) 只有KL 惩罚 𝑅𝑡= −𝛽log 𝜋𝜃(𝑦𝑡|𝑥,𝑦<𝑡) 𝜋ref(𝑦𝑡|𝑦<𝑡) 此时句子还没写完，奖励模型无法打 分。我们只关心这一步有没有“偏离初 心”（KL 散度）。 最后一个token(𝑡= 𝑇) KL 惩罚+RM 总分 𝑅𝑇= −𝛽log 𝜋𝜃(𝑦𝑡|𝑥, 𝑦<𝑡) 𝜋ref(𝑦𝑡|𝑦<𝑡) + 奖励模型给的分数 句子结束（遇到<eos_token> ），奖励模 型RM 终于看完了整句话，给出一个得 分，叠加在最后一步上。 2. 为什么要...

5. 12.2 即时奖励𝑅𝑡的计算 185 • 依靠GAE：把最后的稀疏奖励“抹匀”分摊给前面的每一个动作，让前面的词也能获得梯度更新。 12.3 PPO 微调LLM 实战 应用场景：在电商场景中，很多商品由于没有人买而变成了“长尾商品”。而我们需要为这些长尾商 品“自动”生成一些正向评论，来吸引买家。所以需要训练一个能够编写正向评论的LLM。 1. 首先对一个预训练LLM（gpt2, Qwen2.5-0.5B 等）进行SFT，让LLM 可以编写商品评论。这 里的问题在于这个LLM 虽然能够写电商评论，但写出来的可能是正向评论也可能是负向评论。 2. 训练一个奖励模型（Reward Model），可以针对正向评论打高分，针对负向评论打低分。 3. 使用奖励模型给SFT 后的LLM 输出的电商评论打分，如果分数高（说明是正向评论），则鼓励 SFT 后的LLM 输出这条评论（提升输出这条评论的概率）。如果分数低（说明是负向评论），则 抑制SFT 模型输出这条评论的概率。 12.3.1 监督微调（SFT） 我们的目标是训练一个可以写电商评论的大语言模型。所以我们对gpt2 进行微调，让它可以...

6. 186 Chapter 12. 使用PPO 微调大语言模型--复刻InstructGPT 24 } 25 26 tokenized_dataset_train = ds_train.map(tokenize, **map_kwargs) 27 28 tokenized_dataset_train.set_format(type="torch") 29 # 将eos_token设置为pad_token 30 tokenizer.eos_token = tokenizer.pad_token 31 # 将mlm设置为False，那么数据会被整理为因果注意力模型使用的格式， 32 # 也就是预测下一个token任务需要的数据格式 33 data_collator = DataCollatorForLanguageModeling( 34 tokenizer, 35 mlm=False 36 ) 37 38 dataloader_params = { 39 "batch_size": 2, 40 "collate_fn": data_collator 41 } 42 43 train_d...

7. 12.3 PPO 微调LLM 实战 187 12.3.2 训练奖励模型（Reward Model） 我们会将gpt2 预训练模型微调成一个分类模型（奖励模型）。 我们要训练一个模型作为奖励模型，也就是给模型输入文本，模型可以给一个评分出来。 这 本 书 真 好 看 reward_token GPT2 last_hidden_state out_head reward_head sigmoid 1 评论对应的标签 BCELoss logits 图 12.4 奖励模型训练流程示意图 我们的评论在输入奖励模型之前会先在末尾添加一个reward_token ，作为标记。作用和Bert 用来 训练分类模型时添加的CLS_TOKEN 是一样的。 我们将评论输入gpt2 模型，然后提取gpt2 输出的最后一层隐藏层（last_hidden_state ），并 送入一个我们自己定义的线性层（reward_head ），然后取输出的最后一个元素作为评分。 1 from datasets import load_dataset Python 2 from transformers import Au...

8. 188 Chapter 12. 使用PPO 微调大语言模型--复刻InstructGPT 3 import torch 4 from torch import nn 5 import numpy as np 6 from torch.utils.data import DataLoader 7 from sklearn.metrics import confusion_matrix 8 9 model_path = "./gpt2-chinese-cluecorpussmall" 10 tokenizer = AutoTokenizer.from_pretrained(model_path) 11 12 tokenizer.eos_token = tokenizer.pad_token 13 # reward_token设置为eos_token 14 REWARD_TOKEN_ID = tokenizer.eos_token_id 15 16 ds = load_dataset("csv", data_files="online_shopping_10_cats.csv") 17...

## 4. 与前后章关系

- 在线阅读：https://modelshit-class.vercel.app/02-align/12-ppo-instructgpt/
- 本地对照：`python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 181 --end 202`
