# microGPT

> 原书 PDF 第 17–32 页
> 人话版：[Modelshit Class · microGPT：60 行看懂大模型全貌](https://modelshit-class.vercel.app/01-llm/01-microgpt/)

## 1. 一句话

用 4192 参数的迷你 GPT 串起 embedding、Transformer、训练与推理全流程。

## 2. 章节结构

- 1.2 分词器
- 1.3 自动微分（Autograd）
- 16 numbers
- 1.4 参数
- 1.5 架构
- 1.8 运行
- 1.10 FAQ

## 3. 核心要点

1. 1. microgpt “此文件包含了完整算法，其余一切不过是效率优化。”——Andrej Karpathy 这 200 行代码中运行的数学运算，与 ChatGPT、Claude、Gemini 乃至所有基于 Transformer 架构的语言模型所执行的运算完全一致。区别仅在于规模与速度——而非算法 本身。 以下是整个程序的功能： 1 ┌─────────────────────────────────────────────────────┐ 2 │ 1. DATASET: Load 32,000 human names ("emma", ...) │ 3 │ 2. TOKENIZER: Map each character → integer ID │ 4 │ 3. MODEL: Build a tiny GPT (4,192 parameters) │ 5 │ 4. TRAIN: Show it names, adjust parameters │ 6 │ 5. GENERATE: Ask it to invent new names │ 7 └──────────────...

2. 18 Chapter 1. microgpt 1 emma 2 olivia 3 ava 4 isabella 5 sophia 6 charlotte 7 mia 8 amelia 9 harper 10 ... (~32,000 names follow) 模型的目标是学习数据中的模式，然后生成具有相同统计模式的新文档。作为预览，到脚本结束时， 我们的模型将生成（“幻觉”出！）新的、听起来合理的名字。提前看一下，我们会得到： 1 sample 1: kamon 2 sample 2: ann 3 sample 3: karai 4 sample 4: jaire 5 sample 5: vialan 6 sample 6: karia 7 sample 7: yeran 8 sample 8: anna 9 sample 9: areli 10 sample 10: kaina 11 sample 11: konna 12 sample 12: keylen 13 sample 13: liole 14 sample 14: alerin 15 sample 15: eara...

3. 1.2 分词器 19 在上述代码中，我们收集了数据集中所有不重复的字符（即所有小写字母a-z），将其排序后，每 个字母根据其索引获得一个ID。请注意，这些整数值本身没有任何实际意义；每个标记只是一个独 立的离散符号。它们完全可以被替换为不同的表情符号，而非0、1、2。此外，我们创建了一个名 为BOS （序列起始符）的特殊标记，它作为分隔符使用：告诉模型“新文档在此开始/结束”。在后 续训练中，每个文档两侧都会包裹BOS ：[BOS, e, m, m, a, BOS] 。模型会学习到BOS 标志着一个 新名称的开始，而另一个BOS 则标志着其结束。因此，我们最终得到包含 27 个标记的词汇表（26 个可能的小写字母 a-z，加上 1 个序列起始符标记）。 数据集包含 32,000 个名字。uchars 收集所有名字中的每个独特字符并进行排序，从而得到 字符到整数的映射： 1 Character: a b c d e ... x y z 2 Token ID: 0 1 2 3 4 ... 23 24 25 3 BOS token: 26 BOS （序列起始）是一个特殊token，用于...

4. 20 Chapter 1. microgpt 6 self._local_grads = local_grads 每个Value 存储三项内容：计算结果（data ）、其梯度（grad ，在反向传播过程中填充）以及生成 方式（_children 和_local_grads ）。这些内容共同构成一个计算图——即每项数学运算的记录。 当你写下c = a + b 时，生成的Value 会记住它是通过加法从a 和b 得出的： 1 a (data=3.0) ──┐ 2 ├──(+)──→ c (data=5.0) 3 b (data=2.0) ──┘ 4 5 children: (a, b) 6 local_grads: (1, 1) ← derivative of (a+b) w.r.t. a is 1, 7 derivative of (a+b) w.r.t. b is 1 对于乘法，局部梯度有所不同： 1 def __mul__(self, other): Python 2 return Value(self.data * other.data, (self, other), (ot...

5. 1.3 自动微分（Autograd） 21 1 Forward pass (left to right): compute values 2 ─────────────────────────────────────────────────────────────→ 3 4 a=3.0 ──(×)──→ d=6.0 ──(+)──→ f=7.0 ──(-log)──→ loss=−1.95 5 b=2.0 ──┘ e=1.0 ──┘ 6 7 ←───────────────────────────────────────────────────────────── 8 Backward pass (right to left): compute gradients 9 10 loss.grad = 1.0 11 12 f.grad = 1.0 × (−1/f.data) = −0.143 ← chain rule through -log 13 d.grad = f.grad × 1 = −0.143 ← chain rule through + 14 e.grad = f.gr...

6. 22 Chapter 1. microgpt 20 def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),)) 21 def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),)) 22 def __neg__(self): return self * -1 23 def __radd__(self, other): return self + other 24 def __sub__(self, other): return self + (-other) 25 def __rsub__(self, other): return other + (-self) 26 def __rmul__(self, other): return self * other 27 def __truediv__(self, other): return self * other**-1 28 ...

7. 1.3 自动微分（Autograd） 23 我们从损失节点开始设置self.grad = 1 ，因为𝜕𝐿 𝜕𝐿= 1：损失相对于自身的变化率显然为1。由此， 链式法则只需沿着每条返回参数的路径，将局部梯度相乘即可。 注意这里的+= （累加，而非赋值）。当某个值在计算图中被多处使用（即图存在分支）时，梯度 会沿每个分支独立回流，且必须求和。这是多元链式法则的必然结果：若𝑐通过多条路经影响𝐿，则 总导数等于各路径贡献之和。 在backward() 完成以后，图中的每个Value 都包含一个.grad ，其中含有𝜕𝐿 𝜕𝑣，这告诉我们如果 调整该值，最终损失将如何变化。 以下是一个具体示例。注意a 被使用了两次（图出现分支），因此其梯度是两条路径之和： 1 a = Value(2.0) Python 2 b = Value(3.0) 3 c = a * b # c = 6.0 4 L = c + a # L = 8.0 5 L.backward() 6 print(a.grad) # 4.0 (dL/da = b + 1 = 3 + 1, via both paths) 7 print...

8. 24 Chapter 1. microgpt 7 state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)} 8 for i in range(n_layer): 9 state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd) 10 state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd) 11 state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd) 12 state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd) 13 state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd) 14 state_dict[f'lay...

## 4. 与前后章关系

- 在线阅读：https://modelshit-class.vercel.app/01-llm/01-microgpt/
- 本地对照：`python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 17 --end 32`
