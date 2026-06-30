# 大模型在干嘛

> 原书 PDF 第 33–36 页
> 人话版：[Modelshit Class · 大模型在干嘛：预测下一个词](https://modelshit-class.vercel.app/01-llm/02-next-token/)

## 1. 一句话

LLM 本质是条件概率 P(下一 token | 上文)，训练数据是输入-目标对。

## 2. 章节结构

- 2.4 模型结构的设计
- 12 x

## 3. 核心要点

1. 2. LLM 简介 2.1 大语言模型要解决的问题是什么？ 大语言模型要解决的问题是预测下一个token（next token prediction，NTP）。 LLM 君不见黄河 之 提示词（prompt） . . . LLM 君不见黄河之 水 LLM 君不见黄河之水 天 图 2.1 预测下一个token 2.2 如何训练一个能预测下一个token 的模型？ • 训练数据（输入-目标对）长什么样子？ • 模型结构如何设计？神经网络本质上是一个函数，这个函数长什么样子？参数有多少？ • 神经网络接收输入之后，输出和目标之间的损失（差异）怎么衡量？也就是说损失函数怎么设计？ • 如何让损失函数最小化？梯度下降法以及变种：SGD，Adam，AdamW，Muon 等优化器

2. 34 Chapter 2. LLM 简介 • 梯度下降法需要求损失函数对于参数的导数（梯度），那么梯度如何得来？反向传播算法求参数 的梯度：loss.backward() 。 2.3 训练数据（输入-目标对）长什么样子？ 文本数据：“君不见黄河之水天上来” 输入 预测目标 君不见黄河 之 君不见黄河之 水 君不见黄河之水 天 君不见黄河之水天 上 君不见黄河之水天上 来 表 2.1 输入和预测目标 可以看到训练数据可以通过程序处理原始文本数据，自动生成，所以叫做“自监督学习”。 也就是说，预测目标不是人类标注而来，而是程序通过切分文本自动标注的。 为了充分利用GPU 的并行计算能力，训练数据一般如下组织。 君 不 见 黄 河 之 不 见 黄 河 上下文长度（滑动窗口） 图 2.2 数据组织方法 2.4 模型结构的设计 Decoder-Only Transformer。仅解码器的Transformer 架构。

3. 2.4 模型结构的设计 35 分词后的文本 Token嵌入层 位置嵌入层 Dropout 文本 LayerNorm 1 Multi-head attention Dropout LayerNorm 2 Feed Forward Dropout + + Final LayerNorm Linear output layer 处理输出 输出的文本 12 x GPT-2 (2019) 图 2.3 GPT-2 架构

4. 36 Chapter 2. LLM 简介 2.5 损失函数的设计 LLM 本质上是一个分类模型。 从极大似然估计的角度看到LLM。 假设训练数据为“abcd”，那么我们希望下面的概率越大越好。 𝑃𝜃(d|abc) · 𝑃𝜃(c|ab) · 𝑃𝜃(b|a) (2.1) 其中𝜃是神经网络的参数。最大化上面的式子等价于最大化下面的式子 log{𝑃𝜃(d|abc) · 𝑃𝜃(c|ab) · 𝑃𝜃(b|a)} = log 𝑃𝜃(d|abc) + log 𝑃𝜃(c|ab) + log 𝑃𝜃(b|a) (2.2) 而最大化上面的式子，等价于最小化下面的式子 −log 𝑃𝜃(d|abc) −log 𝑃𝜃(c|ab) −log 𝑃𝜃(b|a) (2.3) 而上面的这个式子就是交叉熵损失函数！ 也就是给定输入“ab”，我们希望预测的下一个token 属于分类“c”的概率越大越好。 给定输入“abc”，我们希望预测的下一个token 属于分类“d”的概率越大越好。 训练数据为"abc" 独热编码 a b c ab LLM 预测目标 下一个token为c的 概率为0.1 训练数据中 预测目标为 toke...

## 4. 与前后章关系

- 在线阅读：https://modelshit-class.vercel.app/01-llm/02-next-token/
- 本地对照：`python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 33 --end 36`
