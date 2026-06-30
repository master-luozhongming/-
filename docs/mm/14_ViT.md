# Vision Transformer

> 原书 PDF 第 241–254 页
> 人话版：[Modelshit Class · Vision Transformer](https://modelshit-class.vercel.app/03-mm/14-vit/)

## 1. 一句话

图像切 Patch 当 token，用 Transformer 做图像分类。

## 2. 章节结构

- 14.2 补丁嵌入（Patch Embedding）
- 14.3 类别对应的 token 和位置编码
- 2 1
- 4 3
- 14.4 注意力头
- 14.5 多头注意力
- 14.7 Vision Transformer
- 14.9 加载MNIST 数据集
- 14.10 训练循环
- 14.11 评估模型

## 3. 核心要点

1. 14. Vision Transformer 提示 ViT: Vision Transformer 基于自注意力的Transformer 模型由Vaswani 等人在2017 年的论文《Attention Is All You Need》中首次提出，并已广泛应用于自然语言处理中。Transformer 模型是OpenAI 用来 创建ChatGPT 的模型。Transformer 不仅适用于文本，还适用于图像，基本上可以处理任何序 列数据。2021 年，Dosovitsky 等人在他们的论文《An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale》中引入了将Transformer 用于计 算机视觉任务（例如图像分类）的想法。在他们的论文中，与卷积网络相比，他们的Vision Transformer 模型能够取得出色的结果，并且需要更少的资源来训练。 在本教程中，我们将从头开始构建一个Vision Transformer 模型，并在MNIST 数据集上进 行测试。 图 14.1 Vit 架构图

2. 242 Chapter 14. Vision Transformer 14.1 导入库和模块 本章所需依赖 Python 1 import torch 2 import torch.nn as nn 3 import torchvision.transforms as T 4 from torch.optim import Adam 5 from torchvision.datasets.mnist import MNIST 6 from torch.utils.data import DataLoader 7 import numpy as np 14.2 补丁嵌入（Patch Embedding） 图 14.2 补丁的例子 补丁嵌入类 Python 1 class PatchEmbedding(nn.Module): 2 def __init__( 3 self, 4 d_model, # 模型的维度 5 img_size, # 图片大小 6 patch_size, # 补丁大小 7 n_channels # 通道数量 8 ): 9 super().__init__() 10...

3. 14.2 补丁嵌入（Patch Embedding） 243 22 23 # B: 批次大小 24 # C: 通道数量 25 # H: 图像高度 26 # W: 图像宽度 27 # P_col: 补丁的列 28 # P_row: 补丁的行 29 def forward(self, x): 30 x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row) 31 x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P) 32 x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model) 33 return x 创建Vision Transformer 的第一步是将输入图像拆分为补丁，并创建这些补丁的线性嵌入序 列。我们能够通过使用 PyTorch 的 Conv2d 方法来实现这一点。 Conv2d 方法获取输入图像，将它们拆分为补丁，并提供大小等于 d_model 的线性投影...

4. 244 Chapter 14. Vision Transformer 1 x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P) Python P_col P_row d_model 1 1 P d_model Flatten 图 14.4 将图像转换为补丁：第二步 最后，我们使用转置方法切换 d_model 和补丁维度，得到 (B, P, d_model) 的形状。 1 x = x.transpose(1, 2) # (B, d_model, P) -> (B, P, d_model) Python P d_model P d_model Transpose 图 14.5 将图像转换为补丁：第三步 14.3 类别对应的 token 和位置编码 位置编码类 Python 1 class PositionalEncoding(nn.Module): 2 def __init__(self, d_model, max_seq_length): 3 super().__init__() 4 # 类别token 5...

5. 14.3 类别对应的 token 和位置编码 245 18 def forward(self, x): 19 # 为批次中的每张图片分配一个类别token 20 tokens_batch = self.cls_token.expand(x.size()[0], -1, -1) 21 # 将类别token添加到每个图像的补丁嵌入数组的开头 22 x = torch.cat((tokens_batch,x), dim=1) 23 # 将位置编码添加到嵌入中 24 x = x + self.pe 25 return x ViT 模型使用向补丁嵌入添加可学习的类别token 的标准方法来执行分类。 1 # class token: 类别或者分类对应的token Python 2 self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) 1 2 3 4 2 1 4 3 图 14.6 位置编码的作用 与LSTM 等按顺序接受嵌入的模型不同，Transformer 并行的接受嵌入。虽然这提高了速度，但 Transformer 不知道序列...

6. 246 Chapter 14. Vision Transformer 8 else: 9 pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model))) 10 # 禁止pe更新，pe保存在了显存中，不会被反向传播更新 11 self.register_buffer('pe', pe.unsqueeze(0)) 在 forward 方法中，输入是多个图像的一批补丁嵌入。例如，32x32 的图像可以分解为 16 个 8x8 大小的补丁。在此 max_seq_length 中，需要为 16+1=17 才能创建足够的位置嵌入，每 个补丁一个，分类对应的 token 一个。 因此，我们需要使用 expand 函数才能使用 self.cls_token 为批处理中的每个图像创建分类 对应的 token 。注意力机制会将有关整个序列的信息编码到序列中的每个token 中。由于每个 token 都受到其自身信息的偏见，因此分类对应的token 会创建序列中所有token 的独立的摘要 信息。 1 def forward(self, x): Python 2...

7. 14.4 注意力头 247 5 6 self.query = nn.Linear(d_model, head_size) 7 self.key = nn.Linear(d_model, head_size) 8 self.value = nn.Linear(d_model, head_size) 9 10 def forward(self, x): 11 # 计算Q, K, V 12 Q = self.query(x) 13 K = self.key(x) 14 V = self.value(x) 15 16 # 𝑄𝐾𝑇 17 attention = Q @ K.transpose(-2,-1) 18 19 # softmax( 𝑄𝐾𝑇 √𝑑head)𝑉 20 attention = attention / (self.head_size ** 0.5) 21 attention = torch.softmax(attention, dim=-1) 22 attention = attention @ V 23 24 return attention MatMul MatMul ...

8. 248 Chapter 14. Vision Transformer Attention (𝑄, 𝐾, 𝑉) = softmax (𝑄𝐾𝑇 √𝑑𝑘 )𝑉 (14.2) 计算注意力的第一步是获取token 的Q、K 和V。token 的Q 是token 要查找的内容，K 是token 包含的内容，V 是token 之间的互信息。Q、K 和V 可以通过线性层传递token 来计算。 1 def forward(self, x): Python 2 # 计算Q，K，V 3 Q = self.query(x) 4 K = self.key(x) 5 V = self.value(x) 我们能够通过计算Q 和K 的点积来获取序列中token 之间的关系。 1 attention = Q @ K.transpose(-2,-1) Python 我们需要缩放这些值以控制初始化时的方差，以便token能够聚合来自多个其他token的信息。通 过将点积除以注意力头大小的平方根来应用缩放。 1 attention = attention / (self.head_size ** 0.5) Pytho...

## 4. 与前后章关系

- 在线阅读：https://modelshit-class.vercel.app/03-mm/14-vit/
- 本地对照：`python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 241 --end 254`
