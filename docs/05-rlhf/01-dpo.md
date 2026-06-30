# 直接偏好优化 (DPO)

## 概述

DPO 是一种无需训练奖励模型的 RLHF 方法，直接从偏好数据中优化策略。

---

## 与 PPO 的对比

| 特性 | PPO | DPO |
|------|-----|-----|
| 奖励模型 | 需要 | 不需要 |
| 训练复杂度 | 高 | 低 |
| 数据需求 | 在线采样 | 离线偏好数据 |
| 稳定性 | 需要调参 | 更稳定 |

---

## 数学推导

### 从 RLHF 到 DPO

传统 RLHF 目标：
$$\max E[R(x, y)] - \beta \cdot KL(\pi_\theta || \pi_{ref})$$

最优策略闭式解：
$$\pi^*(y|x) = \pi_{ref}(y|x) \cdot \frac{\exp(R(x, y)/\beta)}{Z(x)}$$

### 奖励函数表达

$$R(x, y) = \beta \cdot \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \cdot \log Z(x)$$

### DPO 损失函数

$$L(\theta) = -E\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

其中 $y_w$ 是偏好响应，$y_l$ 是非偏好响应。

---

## 代码实现

```python
import torch
import torch.nn as nn

class DPOTrainer:
    def __init__(self, model, ref_model, tokenizer, beta=0.1, lr=1e-6):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def compute_logprobs(self, model, input_ids, labels):
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits[:, :-1, :]
        labels = labels[:, 1:]

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)

        return token_log_probs.mean(dim=-1)

    def dpo_loss(self, prompt, chosen, rejected):
        # 编码
        chosen_inputs = self.tokenizer(prompt + chosen, return_tensors="pt")
        rejected_inputs = self.tokenizer(prompt + rejected, return_tensors="pt")

        # 当前策略的 log 概率
        chosen_logprobs = self.compute_logprobs(
            self.model, chosen_inputs["input_ids"], chosen_inputs["input_ids"]
        )
        rejected_logprobs = self.compute_logprobs(
            self.model, rejected_inputs["input_ids"], rejected_inputs["input_ids"]
        )

        # 参考策略的 log 概率
        with torch.no_grad():
            ref_chosen_logprobs = self.compute_logprobs(
                self.ref_model, chosen_inputs["input_ids"], chosen_inputs["input_ids"]
            )
            ref_rejected_logprobs = self.compute_logprobs(
                self.ref_model, rejected_inputs["input_ids"], rejected_inputs["input_ids"]
            )

        # DPO 损失
        chosen_logratios = chosen_logprobs - ref_chosen_logprobs
        rejected_logratios = rejected_logprobs - ref_rejected_logprobs
        logits = self.beta * (chosen_logratios - rejected_logratios)
        loss = -torch.log(torch.sigmoid(logits)).mean()

        return loss
```

---

## 训练流程

```python
def train_dpo():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    trainer = DPOTrainer(model, ref_model, tokenizer)

    for epoch in range(10):
        for batch in dataloader:
            loss = trainer.dpo_loss(
                batch["prompt"],
                batch["chosen"],
                batch["rejected"]
            )

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
```

---

## 关键要点

1. **无需奖励模型**：直接从偏好数据学习
2. **训练稳定**：避免奖励模型误差传播
3. **实现简单**：只需要前向传播和反向传播
4. **适用场景**：有高质量偏好数据时
