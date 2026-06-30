# 直接偏好优化 (DPO)

## DPO 原理

### 核心思想

DPO (Direct Preference Optimization) 是一种无需训练奖励模型的 RLHF 方法，直接从偏好数据中优化策略。

### 与 PPO 的区别

| 特性 | PPO | DPO |
|------|-----|-----|
| 奖励模型 | 需要 | 不需要 |
| 训练复杂度 | 高 | 低 |
| 数据需求 | 在线采样 | 离线偏好数据 |
| 稳定性 | 需要调参 | 更稳定 |

### 数学公式

DPO 损失函数：

```
L(θ) = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

其中：
- x: prompt
- y_w: preferred response (胜出)
- y_l: rejected response (失败)
- π_θ: 当前策略
- π_ref: 参考策略
- β: 温度参数
- σ: sigmoid 函数

---

## DPO 推导

### 从 RLHF 到 DPO

传统 RLHF 的目标：
```
max E[R(x, y)] - β·KL(π_θ || π_ref)
```

最优策略的闭式解：
```
π*(y|x) = π_ref(y|x) · exp(R(x, y)/β) / Z(x)
```

其中 Z(x) 是归一化常数。

### 奖励函数的表达

从最优策略中解出奖励函数：
```
R(x, y) = β · log(π*(y|x)/π_ref(y|x)) + β · log Z(x)
```

### Bradley-Terry 模型

偏好概率：
```
P(y_w > y_l | x) = σ(R(x, y_w) - R(x, y_l))
```

### DPO 损失

代入奖励函数表达式：
```
P(y_w > y_l | x) = σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))
```

最大化偏好概率等价于最小化负对数似然：
```
L(θ) = -E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

---

## 代码实现

### DPO 训练器

```python
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

class DPOTrainer:
    def __init__(self, model, ref_model, tokenizer, beta=0.1, lr=1e-6):
        """
        DPO 训练器

        Args:
            model: 当前策略模型
            ref_model: 参考模型
            tokenizer: 分词器
            beta: 温度参数
            lr: 学习率
        """
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def compute_logprobs(self, model, input_ids, labels):
        """计算序列的log概率"""
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits[:, :-1, :]
        labels = labels[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)

        # 返回平均log概率
        return token_log_probs.mean(dim=-1)

    def dpo_loss(self, prompt, chosen, rejected):
        """
        计算 DPO 损失

        Args:
            prompt: 输入提示
            chosen: 偏好的response
            rejected: 不偏好的response

        Returns:
            loss: DPO损失
            metrics: 评估指标
        """
        # 编码输入
        chosen_inputs = self.tokenizer(prompt + chosen, return_tensors="pt")
        rejected_inputs = self.tokenizer(prompt + rejected, return_tensors="pt")

        # 计算当前策略的log概率
        chosen_logprobs = self.compute_logprobs(
            self.model,
            chosen_inputs["input_ids"],
            chosen_inputs["input_ids"]
        )
        rejected_logprobs = self.compute_logprobs(
            self.model,
            rejected_inputs["input_ids"],
            rejected_inputs["input_ids"]
        )

        # 计算参考策略的log概率
        with torch.no_grad():
            ref_chosen_logprobs = self.compute_logprobs(
                self.ref_model,
                chosen_inputs["input_ids"],
                chosen_inputs["input_ids"]
            )
            ref_rejected_logprobs = self.compute_logprobs(
                self.ref_model,
                rejected_inputs["input_ids"],
                rejected_inputs["input_ids"]
            )

        # 计算log比率
        chosen_logratios = chosen_logprobs - ref_chosen_logprobs
        rejected_logratios = rejected_logprobs - ref_rejected_logprobs

        # DPO 损失
        logits = self.beta * (chosen_logratios - rejected_logratios)
        loss = -torch.log(torch.sigmoid(logits)).mean()

        # 评估指标
        metrics = {
            "chosen_logratios": chosen_logratios.mean().item(),
            "rejected_logratios": rejected_logratios.mean().item(),
            "logits": logits.mean().item(),
            "accuracy": (logits > 0).float().mean().item()
        }

        return loss, metrics

    def train_step(self, batch):
        """单步训练"""
        self.model.train()

        total_loss = 0
        total_metrics = {}

        for item in batch:
            prompt = item["prompt"]
            chosen = item["chosen"]
            rejected = item["rejected"]

            loss, metrics = self.dpo_loss(prompt, chosen, rejected)

            total_loss += loss
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v

        # 平均损失
        total_loss /= len(batch)
        for k in total_metrics:
            total_metrics[k] /= len(batch)

        # 更新
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item(), total_metrics
```

### 偏好数据格式

```python
# 偏好数据示例
preference_data = [
    {
        "prompt": "什么是机器学习？",
        "chosen": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习，而无需显式编程。",
        "rejected": "机器学习就是让机器自己学习。"
    },
    {
        "prompt": "解释什么是神经网络",
        "chosen": "神经网络是一种受人脑启发的计算模型，由多层相互连接的节点组成，能够学习复杂的模式。",
        "rejected": "神经网络就是电脑里的网络。"
    }
]
```

---

## 使用 DPO 微调大语言模型

### 完整训练流程

```python
def train_dpo():
    # 1. 加载模型
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 2. 创建训练器
    trainer = DPOTrainer(model, ref_model, tokenizer, beta=0.1, lr=1e-6)

    # 3. 准备数据
    dataset = load_preference_data()

    # 4. 训练循环
    for epoch in range(10):
        for batch in dataloader:
            loss, metrics = trainer.train_step(batch)

            print(f"Epoch {epoch}, Loss: {loss:.4f}, "
                  f"Accuracy: {metrics['accuracy']:.4f}")

        # 5. 保存模型
        model.save_pretrained(f"dpo_model_epoch_{epoch}")
```

### 数据加载

```python
def load_preference_data():
    """加载偏好数据"""
    # 从文件加载
    import json
    with open("preference_data.json", "r") as f:
        data = json.load(f)

    # 或者从HuggingFace加载
    # from datasets import load_dataset
    # data = load_dataset("Anthropic/hh-rlhf")

    return data
```

---

## DPO 的优势

### 相比 PPO
1. **无需奖励模型**: 直接从偏好数据学习
2. **训练更稳定**: 避免了奖励模型的误差传播
3. **实现更简单**: 不需要复杂的在线采样
4. **计算成本低**: 只需要前向传播和反向传播

### 适用场景
1. **有高质量偏好数据**: 人工标注的偏好对
2. **计算资源有限**: 无法训练额外的奖励模型
3. **需要快速迭代**: 离线训练更方便

---

## DPO 的局限性

### 数据需求
- 需要高质量的偏好数据
- 数据标注成本高
- 数据质量影响模型性能

### 理论限制
- 假设偏好数据符合 Bradley-Terry 模型
- 可能存在分布偏移问题

### 实践挑战
- β 参数需要仔细调优
- 可能出现过拟合

---

## 超参数调优

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| β (beta) | 0.1 - 0.5 | 温度参数，控制偏离参考策略的程度 |
| lr | 1e-6 - 5e-6 | 学习率 |
| batch_size | 4 - 16 | 批量大小 |
| epochs | 1 - 5 | 训练轮数 |

---

## 与其他算法对比

| 算法 | 数据需求 | 训练复杂度 | 稳定性 | 性能 |
|------|----------|------------|--------|------|
| PPO | 在线采样 | 高 | 中 | 高 |
| DPO | 偏好数据 | 低 | 高 | 高 |
| GRPO | 在线采样 | 中 | 高 | 高 |

---

## 关键要点

1. **DPO 是一种无需奖励模型的 RLHF 方法**
2. **直接从偏好数据中优化策略**
3. **实现简单，训练稳定**
4. **适用于有高质量偏好数据的场景**

---

*参考: 原文档 Chapter 11*
