# DALL-E 2

> 原书 PDF 第 359–400 页
> 人话版：[Modelshit Class · DALL-E 2：文生图全家桶](https://modelshit-class.vercel.app/03-mm/20-dalle2/)

## 1. 一句话

CLIP 语义空间 + 先验 + 解码器，文本到图像的级联 pipeline。

## 2. 章节结构

- 20.2 先验模型（Prior Model）
- 20.2.1 先验模型的训练
- 20.2.2 先验模型的推理
- 20.2.3 关键代码解释
- 1 + 𝑠⋅𝜋
- 1.0 / schedule_values["alphas"])
- 1.0 - schedule_values["alpha_bars"]
- 20.3 条件扩散模型
- 20.3.1 概述
- 20.3.2 残差块
- 20.3.3 注意力块
- 20.3.4 下采样
- 20.3.5 上采样
- 20.3.6 模型的训练
- 20.3.7 损失的计算
- 20.4 训练配置
- 20.5 训练
- 20.6 训练结果

## 3. 核心要点

1. 20. DALL-E2 （文生图） DALL-E2 • 输入：文本提示词 • 输出：图像 训练DALL-E2 的数据集 图文对！ 去噪扩散概率模型（DDPM）是一种流行的生成式人工智能模型，由Ho 等人于2020 年提出，并由 Nichol 等人于2021 年对其进行了改进。这些模型背后的基本思想是，在正向扩散过程中将噪声 添加到图像中，以便训练模型预测在反向扩散过程中应在特定时间步去除的噪声。在对图像进行采 样（去噪）时，需要从纯噪声的图像开始，并在每个时间步迭代地去除模型预测的噪声，直到获得 最终图像。 为了让DDPM 生成多种类型的图像，同时仍允许用户选择所需的图像类型，模型需要根据某些输 入进行条件调节（条件扩散模型）。Ramesh 等人提出了一种名为unCLIP 的条件调节方法，该方法 已用于OpenAI 的DALL-E 2 模型。在Ramesh 等人描述的方法中，输入的图片标题或者描述首先 被传递到一个先验网络（prior model），该网络将使用经过训练的CLIP 模型来获取CLIP 文本 嵌入。然后，仅解码器架构的Transformer 使用这些文本嵌入来生成可...

2. 360 Chapter 20. DALL-E2 （文生图） 图 20.1 DALL-E2 模型整体架构 DALL-E2 需要训练3 个模型 1. CLIP 模型 2. 先验模型：Decoder-Only Transformer，先验模型的目标是生成高质量的条件。 3. 条件扩散模型：UNet 20.1 CLIP 为了从文本创建扩散图像（文生图），我们将使用CLIP 模型生成的嵌入（文本特征向量，图像特征 向量）。从CLIP 获得的文本嵌入用于调节先验模型，使其扩散相应的图像嵌入。然后，这些图像嵌 入用于调节解码器模型，用来指导解码器生成对应的图像。 提示 CLIP 模型训练完毕后，冻结起来供后续使用。 20.2 先验模型（Prior Model） 危险 训练先验模型时，需要使用上一节训练的CLIP 模型，CLIP 模型必须冻结！

3. 20.2 先验模型（Prior Model） 361 图 20.2 先验模型架构图 先验模型的作用：根据文本标题预测CLIP 图像嵌入 。也可以放弃先验模型，而以CLIP 文本嵌入 为条件，而不是用先验模型生成的CLIP 图像嵌入作为条件，但使用先验模型的效果最佳。 先验模型是一个仅解码器架构的Transformer。 20.2.1 先验模型的训练 仅解码器Transformer 因果注意力掩码 文本标题 CLIP文本特征 向量 时间步嵌入 加噪声的CLIP 图像特征向量 可学习嵌入 可学习嵌入 MLP 预测的CLIP图像特征向量 真实的CLIP图像特征向量 Loss 图 20.3 先验模型的训练原理 要训练一个模型，我们必须搞清楚模型的输入和输出（预测目标）。 训练先验模型时，我们手上有的是：图文对和一个训练好的CLIP 模型。 先验模型的输入有6 个： 1. Text Captions：图文对中的“文本”。 2. CLIP Text Embeddings：图文对中的“文本”经过上一节训练好的冻结的CLIP 模型生成 的CLIP 文本嵌入。 3. Timestep Embedd...

4. 362 Chapter 20. DALL-E2 （文生图） 4. Noisy Image Embeddings：添加了𝑡步噪声的CLIP 图像嵌入。这个CLIP 图像嵌入是图文 对中的“图像”经过上一节训练好的冻结的CLIP 模型生成的CLIP 图像嵌入。 5. Learned Embeddings：一个随机初始化的可学习的嵌入。 6. Causal Attention Mask：因果注意力掩码。因为我们要训练的是一个Decoder-Only Transformer 架构的模型，所以需要使用因果注意力掩码。 危险 Noisy Image Embeddings 是针对CLIP 图片嵌入添加噪声，而不是针对图片添加噪声！ 先验模型的输出（预测目标）： • 图文对中的“图像”经过上一节训练好的冻结的CLIP 模型生成的CLIP 图像嵌入。 训练细节： 1. 6 个输入经过Decoder-Only Transformer 之后会输出5 个张量，我们取最后一个，也就 是图中蓝色的Learned Embeddings 。 2. Learned Embeddings 会送入一个MLP，输出的就...

5. 20.2 先验模型（Prior Model） 363 提示 所以只有Text Captions 是我们自己编写的文本提示词，其它输入都是模型或者程序自动 生成的。 先验模型的输出： 1. 6 个输入经过Decoder-Only Transformer 之后会输出5 个张量，我们取最后一个，也就 是图中蓝色的Learned Embeddings 。 2. Learned Embeddings 会送入两次MLP，生成两个预测的CLIP 图片嵌入。 3. 两个预测的CLIP图片嵌入分别和CLIP文本嵌入计算余弦相似度，然后选取最相似的一个作为 预测。 20.2.3 关键代码解释 20.2.3.1 余弦调度的方差计划 图 20.5 线性调度方差和余弦调度方差的对比 对于方差调度，在Ho 等人的原始DDPM 论文中，采用了线性调度。虽然该调度在高分辨率图像上 效果良好，但当图像较小（64x64 或更小）时，前向过程最终会产生过多的噪声。为了解决这个问 题，Nichol 等人建议使用余弦调度，其公式如下所示。 𝛽𝑡= 1 −𝛼𝑡 𝛼𝑡−1 𝛼𝑡= 𝑓(𝑡) 𝑓(0) 𝑓(𝑡) = cos (...

6. 364 Chapter 20. DALL-E2 （文生图） 11 t = torch.linspace(0, max_time, max_time + 1) 12 a_bars = torch.cos( 13 (((t / max_time) + s) / (1 + s)) * (np.pi / 2) 14 ) ** 2 15 a_bars = a_bars / a_bars[0] 16 betas = 1 - (a_bars[1:] / a_bars[:-1]) 17 betas = torch.clamp(betas, min=0, max=0.999) 18 else: 19 Exception("Beta schedule not implemented.") 20 21 return betas 22 23 def get_schedule_values(config): 24 schedule_values = {} 25 schedule_values["betas"] = get_beta_schedule( 26 config.prior.schedule, 2...

7. 20.2 先验模型（Prior Model） 365 61 # 返回x_t, 和添加的噪声 62 return x_noisy, noise 20.2.3.2 时间步嵌入 时间步嵌入是扩散的重要组成部分。这是因为不同时间步的图像具有不同的噪声量。为了在我们的 模型中利用这些信息，我们将使用正弦位置编码。这些位置编码与Transformer 中常用的位置编 码相同。主要区别在于，我们的输入时间步很可能不会按顺序排列，并且包含所有可能的时间步，因 此我们只需要获取与输入时间步对应的位置编码。 1 class SinusoidalPositionalEncodings(nn.Module): Python 2 def __init__( 3 self, 4 max_seq_length, # 最大序列长度 5 width # 模型的宽度（嵌入的维度d_model） 6 ): 7 super().__init__() 8 # Create positional encodings 9 pe = torch.zeros(max_seq_length, width) 10 for pos i...

8. 366 Chapter 20. DALL-E2 （文生图） 20.2.3.3 冻结模型 先验模型首先将图像的文本标题作为输入，然后获取CLIP 文本嵌入和图像嵌入。加载CLIP 模型 时，所有层都应冻结，并将模式设置为eval。 1 def freeze_model(model, set_eval=True): Python 2 if set_eval: 3 model.eval() 4 5 for param in model.parameters(): 6 param.requires_grad = False 7 8 # Constructor 9 self.clip = CLIP(config).to(config.device) 10 self.clip.load_state_dict(torch.load( 11 config.clip.model_location, 12 map_location=config.device 13 )) 14 # 冻结clip模型 15 freeze_model(self.clip) 16 17 # Forward 18 image...

## 4. 与前后章关系

- 在线阅读：https://modelshit-class.vercel.app/03-mm/20-dalle2/
- 本地对照：`python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 359 --end 400`
