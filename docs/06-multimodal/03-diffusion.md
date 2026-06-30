# 扩散模型

## 概述

扩散模型通过逐步添加和去除噪声来生成图像。

---

## 核心思想

```
正向扩散：图像 → 加噪 → 加噪 → ... → 纯噪声
反向扩散：纯噪声 → 去噪 → 去噪 → ... → 生成图像
```

---

## 正向扩散过程

### 数学表达

$$q(x_t | x_{t-1}) = N(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

### 代码实现

```python
import torch

def forward_diffusion(x_0, t, betas):
    """正向扩散：添加噪声"""
    noise = torch.randn_like(x_0)

    alpha = 1 - betas
    alpha_bar = torch.cumprod(alpha, dim=0)

    # 直接采样 x_t
    alpha_bar_t = alpha_bar[t].reshape(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

    return x_t, noise
```

---

## 反向扩散过程

### 噪声预测网络

```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # 编码器
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)

        # 瓶颈层
        self.bottleneck = self._block(256, 512)

        # 解码器
        self.dec3 = self._block(512 + 256, 256)
        self.dec2 = self._block(256 + 128, 128)
        self.dec1 = self._block(128 + 64, 64)

        # 输出层
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, t):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        # 瓶颈
        b = self.bottleneck(F.max_pool2d(e3, 2))

        # 解码
        d3 = self.dec3(torch.cat([F.interpolate(b, scale_factor=2), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))

        return self.out(d1)
```

---

## 训练过程

```python
def train_diffusion(model, dataloader, betas, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for images in dataloader:
            # 随机时间步
            t = torch.randint(0, len(betas), (images.shape[0],))

            # 正向扩散
            x_t, noise = forward_diffusion(images, t, betas)

            # 预测噪声
            predicted_noise = model(x_t, t)

            # 计算损失
            loss = F.mse_loss(predicted_noise, noise)

            # 更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 采样过程

```python
def sample(model, betas, shape):
    """从噪声生成图像"""
    x = torch.randn(shape)

    for t in reversed(range(len(betas))):
        # 预测噪声
        predicted_noise = model(x, torch.tensor([t]))

        # 去噪
        alpha = 1 - betas[t]
        alpha_bar = torch.cumprod(alpha, dim=0)[t]

        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise
        ) + torch.sqrt(betas[t]) * noise

    return x
```

---

## DDPM 超参数

```python
# 噪声调度
num_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps)
```

---

## 条件生成

### 文本条件

```python
class ConditionalUNet(UNet):
    def __init__(self, text_embed_dim=512):
        super().__init__()
        self.text_embed = nn.Linear(text_embed_dim, 512)

    def forward(self, x, t, text_embedding):
        text_cond = self.text_embed(text_embedding)
        # 将文本条件注入到网络中
        ...
```

---

## 关键要点

1. **正向扩散**逐步添加噪声
2. **反向扩散**逐步去除噪声
3. **U-Net** 是常用的噪声预测网络
4. **条件生成**可以控制生成内容
