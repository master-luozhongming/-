# CLIP

> 原书 PDF 第 255–272 页
> 人话版：[Modelshit Class · CLIP：让图文站到同一空间](https://modelshit-class.vercel.app/03-mm/15-clip/)

## 1. 一句话

双塔编码器 + 对比学习，图文共享嵌入空间，支持零样本分类。

## 2. 章节结构

- 15.2.1 位置嵌入
- 15.2.2 注意力头
- 15.2 图像和文本编码器
- 15.2.3 多头注意力
- 15.2.4 Transformer 编码器
- 15.2.5 文本的分词器
- 69 88
- 15.2.6 文本编码器
- 15.2.7 图像编码器
- 15.3 CLIP 模型
- 15.3.1 数据
- 15.3.2 训练参数
- 15.3.3 加载数据集
- 15.3.4 训练模型
- 15.3.5 评估模型
- 100.0 * image_features @ text_features.T
- 15.3.6 零样本分类

## 3. 核心要点

1. 15. CLIP 计算机视觉系统历来仅限于一组固定的类别，CLIP 是一场革命，它允许通过“预测哪些图像和文 本配对在一起”来识别开放世界中的对象。CLIP 能够通过学习批量训练数据的图像和文本特征之 间的余弦相似度来预测这一点。这在下图的对比预训练部分显示，其中图像之间的点积特征 {𝐼1, 𝐼2, …, 𝐼𝑁} 和文本特征 {𝑇1, 𝑇2, …, 𝑇𝑁} 被占用。 图 15.1 clip 原理，来自原始论文 在本章中，我们将从头开始构建CLIP 并在MNIST 数据集上对其进行测试。 15.1 导入库和模块 1 import torch Python 2 import torch.nn as nn 3 import torch.optim as optim 4 import torchvision.transforms as T 5 from torch.utils.data import Dataset, DataLoader 6 from datasets import load_dataset 7 import matplotlib.pyplot as plt 8 imp...

2. 256 Chapter 15. CLIP 15.2 图像和文本编码器 我们将首先构建图像和文本编码器。两者分别将图像和文本嵌入到单个token 中，然后可用于对比 损失计算。 15.2.1 位置嵌入 1 class PositionalEmbedding(nn.Module): Python 2 def __init__(self, width, max_seq_length): 3 super().__init__() 4 # 创建一个 (token的数量, 嵌入的维度) 形状的全0张量 5 # width 就是 d_model 6 pe = torch.zeros(max_seq_length, width) 7 # 将位置编码信息填充到pe中 8 for pos in range(max_seq_length): 9 for i in range(width): 10 if i % 2 == 0: 11 pe[pos][i] = np.sin(pos/(10000 ** (i/width))) 12 else: 13 pe[pos][i] = np.cos(pos/(100...

3. 15.2 图像和文本编码器 257 21 attention = attention @ V 22 return attention Transformer 编码器和解码器之间的主要区别在于解码器使用注意力掩码，而编码器则不使用。 虽然CLIP 是仅编码器模型（Encoder-Only），但由于在分词时，对输入文本添加了pad（填充 符），所以仍然需要与文本编码器一起使用掩码。请注意，掩码是可选的，因此这个注意力头可用于 文本和视觉编码器。 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 注意力分数 1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 1 1 1 0 0 掩码 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 掩码注意力分数 - - - - - - - - - - 掩码填充 图 15.2 注意力分数的掩码 1 # 使用注意力掩码 Python 2 if mask is not None: 3 attention = attention.masked_fill(mask == ...

4. 258 Chapter 15. CLIP 8 self.ln1 = nn.LayerNorm(width) 9 10 # 多头注意力 11 self.mha = MultiHeadAttention(width, n_heads) 12 13 # 层归一化 14 self.ln2 = nn.LayerNorm(width) 15 16 # MLP 17 self.mlp = nn.Sequential( 18 nn.Linear(self.width, self.width*r_mlp), 19 nn.GELU(), 20 nn.Linear(self.width*r_mlp, self.width) 21 ) 22 23 def forward(self, x, mask=None): 24 x = x + self.mha(self.ln1(x), mask=mask) 25 x = x + self.mlp(self.ln2(x)) 26 return x 15.2.5 文本的分词器 我们的词汇表是ascii 码，所以 vocab_size = 256 。 1 def to...

5. 15.2 图像和文本编码器 259 T E X T T E X T <SOT> <EOT> T E X T <SOT> <EOT> <PAD> <PAD> <PAD> <PAD> 84 84 2 3 0 69 88 0 0 0 1 1 1 1 1 1 0 0 0 0 输出： 添加开始和结束token： 添加填充符： UTF编码： 掩码： 图 15.3 分词过程 分词器的第一步是将文本的开头和文本的结尾token 添加到输入字符串中。 1 text = chr(2) + text + chr(3) Python 添加文本的开头和文本的结尾token 后，我们需要将序列的长度填充到最大序列长度。 1 text = text + "".join([chr(0) for _ in range(10-len(text))]) Python 我们通过将文本序列编码为UTF-8 并将输出转换为 IntTensor 来完成分词。 1 text = torch.IntTensor(list(text.encode("utf-8"))) Python 对文本进行分词后，我们需要为文本创建掩码。虽然T...

6. 260 Chapter 15. CLIP 13 self.encoder_embedding = nn.Embedding(vocab_size, width) 14 self.positional_embedding = PositionalEmbedding( 15 width, 16 max_seq_length 17 ) 18 self.encoder = nn.ModuleList([ 19 TransformerEncoder(width, n_heads) 20 for _ in range(n_layers) 21 ]) 22 # 可学习投影（projection） 23 # 𝑊width × emb_dim 24 self.projection = nn.Parameter(torch.randn(width, emb_dim)) 25 26 def forward(self, text, mask=None): 27 # 文本嵌入 28 x = self.encoder_embedding(text) 29 # 位置嵌入 30 x = self.positio...

7. 15.2 图像和文本编码器 261 1 for encoder_layer in self.encoder: Python 2 x = encoder_layer(x, mask=mask) 编码器层的输出是文本的特征。我们将使用从 EOT 的嵌入中抽取的特征。 1 # 从EOT的嵌入抽取特征 Python 2 x = x[torch.arange( 3 text.shape[0]), 4 torch.sub(torch.sum(mask[:,0],dim=1),1) 5 ] 最后，我们通过计算特征和投影之间的点积，将文本嵌入映射到CLIP 嵌入空间中，并通过除以归 一化的点积对其进行归一化。 为什么要做映射？ 主要是为了在CLIP 嵌入空间中，文本嵌入向量的维度和图像嵌入向量的维度一致。 向量除 以向量的模长，就是模长为1 的向量。这样文本嵌入向量和图像嵌入向量都变成了模长为1 的 向量。两个向量的点积，就是两个向量的余弦相似度！ 1 if self.projection is not None: Python 2 x = x @ self.projection 3 x = x...

8. 262 Chapter 15. CLIP 25 width, 26 kernel_size=patch_size, 27 stride=patch_size 28 ) 29 self.cls_token = nn.Parameter(torch.randn(1, 1, width)) 30 self.positional_embedding = PositionalEmbedding( 31 width, 32 self.max_seq_length 33 ) 34 self.encoder = nn.ModuleList([ 35 TransformerEncoder(width,n_heads) 36 for _ in range(n_layers) 37 ]) 38 39 # 𝑊width × emb_dim 40 self.projection = nn.Parameter(torch.randn(width, emb_dim)) 41 42 def forward(self,x): 43 # 补丁嵌入 44 x = self.linear_project(x) 45 x...

## 4. 与前后章关系

- 在线阅读：https://modelshit-class.vercel.app/03-mm/15-clip/
- 本地对照：`python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 255 --end 272`
