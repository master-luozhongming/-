# ClipCap

> 原书 PDF 第 273–286 页
> 人话版：[Modelshit Class · ClipCap：看图说话](https://modelshit-class.vercel.app/03-mm/16-clipcap/)

## 1. 一句话

CLIP 视觉特征 + GPT 文本解码，实现图像描述生成。

## 2. 章节结构

- 16.3 推理
- 16.4.1 项目配置
- 16.4.2 处理数据
- 16.4 ClipCap 的代码实现
- 16.4.3 模型结构
- 16.4.4 准备训练数据集
- 16.4.5 训练
- 16.4.6 推理

## 3. 核心要点

1. 16. ClipCap（图生文） Image Captioning Model 图片标题模型（image captioning model）是一种将图片作为输入并生成图片描述的模型。 下面是一个简单的示例，展示了一个图片标题模型： 图片标题模型(Image Captioning Model) "A drawing of a blue car" Image Captioning Model 图 16.1 图片标题模型的简单示例 16.1 ClipCap 的工作原理 ClipCap 是一种结合了 CLIP 和 GPT-2 的图片标题架构。 CLIP 是我们将用来创建输入图像嵌入的模型。 GPT-2 是一种基于解码器的模型，用于生成文本。 ClipCap 的基本工作原理如下： 输入图像首先通过 CLIP 模型转换为嵌入，目的是利用这种嵌入（捕捉图像意义）来引导 GPT-2 生成文本。 但有一个问题：CLIP 和 GPT-2 的嵌入空间不同。所以我们不能直接把这个嵌入输入到 GPT-2 中。 为了解决这个问题，我们使用一个映射网络将 CLIP 嵌入映射到 GPT-2 的嵌入空间。 这些映...

2. 274 Chapter 16. ClipCap（图生文） 输入图片 C L I P Mapping Network 嵌入 G P T 2 "A drawing of a blue car" 标题 将图片的嵌入从 CLIP的嵌入空间 映射到GPT2的嵌 入空间 我们将使用这些嵌入作为“前缀” (prefix) 来使得GPT2生成标题 但问题是CLIP和GPT2有着不同的嵌入 空间，所以我们必须将CLIP嵌入映射到 GPT2的嵌入空间，才能使用这些嵌入 图 16.2 将图片的CLIP 嵌入映射到GPT2 的嵌入空间 16.2 关于训练 CLIP 生成的图像嵌入开箱即用已经足够好——所以我们不训练 CLIP 模型。 根据 GPT-2 是否经过微调，ClipCap 有两种变体 ： • 如果我们对 GPT-2 进行微调 ，那么我们就用 MLP 作为映射网络。GPT-2 和 MLP 都经过 训练。 • 如果我们不对 GPT-2 进行微调，那么我们就用Transformer 架构（例如Bert）作为映射网 络，只有Transformer 本身被训练。 就我而言，我选择对 GPT-2 模型进行微...

3. 16.3 推理 275 输入图片 C L I P Mapping Network ClipCap的推理 嵌入 t=1时 Prefix Embeddings GPT-2 “A” 预测下一个token Prefix Embeddings GPT-2 “blue” 预测下一个token t=2时 "A" Prefix Embeddings GPT-2 “car” 预测下一个token t=3时 "A" "blue" Prefix Embeddings GPT-2 <EOS> 预测下一个token t=4时 "A" "blue" "car" 图 16.3 ClipCap 的推理过程 16.4 ClipCap 的代码实现 16.4.1 项目配置 config.py Python 1 import torch 2 3 CLIP_MODEL_PATH = "./chinese-clip-vit-base-patch16" 4 # 一张图片的嵌入经过投影转换成10个token的embedding，每个embedding的dim是768 5 IMAGE_TOKEN_LENGTH = 10 # 图片...

4. 276 Chapter 16. ClipCap（图生文） 7 def main(): 8 # 加载clip模型 9 # clip模型只用来生成图片的嵌入，不进行微调。 10 clip_model = ChineseCLIPModel.from_pretrained(CLIP_MODEL_PATH) 11 # 加载clip处理器 12 processor = ChineseCLIPProcessor.from_pretrained(CLIP_MODEL_PATH) 13 # 将2张图片进行处理，处理完之后交给clip抽取特征 14 inputs_1 = processor(images=Image.open("1.jpg"), return_tensors="pt") 15 inputs_2 = processor(images=Image.open("2.jpg"), return_tensors="pt") 16 # 获取第一张图片的嵌入（dim: 512） 17 image_1_features = clip_model.get_image_features(**inputs...

5. 16.4 ClipCap 的代码实现 277 16.4.3 模型结构 图片 clip的图像编码器 图片嵌入（维度：512） 投影 10个嵌入（每个嵌入维度：768） 将图片嵌入投影成和词嵌入相同的形状， 相当于10个token 图片描述文本的词嵌入（每个嵌入维度：768） gpt2 注意：只使用了 clip的图像编码 器！ 图片描述文本 转换成gpt2的词嵌入 图 16.4 clipcap 的模型设计要点 model.py Python 1 import torch 2 import torch.nn as nn 3 from transformers import GPT2LMHeadModel 4 5 from typing import Sequence 6 from config import LLM_PATH, IMAGE_TOKEN_LENGTH, IMAGE_EMBD_DIM 7 8 9 class MLP(nn.Module): 10 """投影层""" 11 def __init__(self, sizes: Sequence[int]): 12 super()...

6. 278 Chapter 16. ClipCap（图生文） 18 19 def forward(self, x: torch.Tensor) -> torch.Tensor: 20 x = x.float() 21 x = self.l1(x) 22 x = self.act1(x) 23 x = self.l2(x) 24 return x 25 26 27 class ClipCaptionModel(nn.Module): 28 def __init__(self): 29 super(ClipCaptionModel, self).__init__() 30 # 大语言模型：用来生成图片的文本描述 31 self.gpt2 = GPT2LMHeadModel.from_pretrained(LLM_PATH) 32 # gpt2的词嵌入维度是768 33 self.word_embd_dim = self.gpt2.config.n_embd 34 # 投影层定义 35 self.projection = MLP(( 36 # 输入维度是512,也就是clip的图像编码器输...

7. 16.4 ClipCap 的代码实现 279 16.4.4 准备训练数据集 clipcap_dataset.py Python 1 import torch 2 from torch.utils.data import Dataset 3 import pickle 4 from typing import Tuple 5 from config import IMAGE_TOKEN_LENGTH, MAX_LENGTH 6 7 8 class ClipCapDataset(Dataset): 9 def __init__(self, tokenizer): 10 # 填充符 11 pad_id = tokenizer.pad_token_id 12 # 取出图片的文本和图片的嵌入 13 with open("caption_image.pkl", 'rb') as f: 14 caption_list, image_id2embed = pickle.load(f) 15 print('图片嵌入的总数:{}'.format(len(image_id2embed))) 16 pr...

8. 280 Chapter 16. ClipCap（图生文） 48 mask = torch.tensor(mask).long() 49 50 image_embed_list.append(image_embed) 51 caption_ids_list.append(caption_ids) 52 mask_list.append(mask) 53 # 保存训练数据 54 with open("train_data.pkl", 'wb') as f: 55 pickle.dump([ 56 image_embed_list, # clip输出的图片特征的列表 57 caption_ids_list, # 图片文本的input_ids的列表 58 mask_list # 掩码的列表 59 ], f) 60 self.image_embed_list = image_embed_list 61 self.caption_ids_list = caption_ids_list 62 self.mask_list = mask_list 63 print(f'训练数据总数：{len(s...

## 4. 与前后章关系

- 在线阅读：https://modelshit-class.vercel.app/03-mm/16-clipcap/
- 本地对照：`python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 273 --end 286`
