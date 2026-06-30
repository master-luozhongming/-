# 扩散模型

## 为什么需要扩散模型？

生成模型的目标：从噪声中生成真实的图像。

之前的生成模型：
- **GAN**：生成对抗网络，训练不稳定
- **VAE**：变分自编码器，生成图像模糊
- **Flow**：标准化流，计算开销大

**扩散模型**（Diffusion Model）通过**逐步添加和去除噪声**来生成图像，效果更好，训练更稳定。

---

## 核心思想

### 正向扩散

将图像逐步添加噪声，直到变成纯噪声：

```
图像 → 加噪 → 加噪 → ... → 纯噪声
x_0 → x_1 → x_2 → ... → x_T
```

### 反向扩散

从纯噪声中逐步去除噪声，恢复出图像：

```
纯噪声 → 去噪 → 去噪 → ... → 生成图像
x_T → x_{T-1} → x_{T-2} → ... → x_0
```

### 直观理解

想象你在雕刻：
- **正向扩散**：逐渐往雕像上涂泥，直到变成泥块
- **反向扩散**：逐渐从泥块中雕刻出雕像

---

## 正向扩散过程

### 数学表达

正向扩散是一个**马尔可夫链**：

$$q(x_t | x_{t-1}) = N(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

其中：
- $\beta_t$：噪声调度（noise schedule），控制每步添加多少噪声
- $N(\mu, \sigma^2)$：正态分布

### 直观理解

每一步：
1. 将上一步的图像乘以 $\sqrt{1-\beta_t}$（缩小）
2. 添加方差为 $\beta_t$ 的噪声

### 重参数化技巧

我们可以直接从 $x_0$ 采样 $x_t$，不需要逐步计算：

$$q(x_t | x_0) = N(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

其中：
- $\alpha_t = 1 - \beta_t$
- $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$

### 代码实现

```python
import torch

def forward_diffusion(x_0, t, betas):
    """
    正向扩散：添加噪声

    参数:
        x_0: 原始图像
        t: 时间步
        betas: 噪声调度
    """
    # 生成噪声
    noise = torch.randn_like(x_0)

    # 计算 alpha 和 alpha_bar
    alpha = 1 - betas
    alpha_bar = torch.cumprod(alpha, dim=0)

    # 直接采样 x_t
    alpha_bar_t = alpha_bar[t].reshape(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

    return x_t, noise
```

### 噪声调度

```python
# 线性噪声调度
num_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps)
```

---

## 反向扩散过程

### 目标

反向扩散的目标是学习：

$$p_\theta(x_{t-1} | x_t) = N(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

其中 $\mu_\theta(x_t, t)$ 是神经网络预测的均值。

### 关键洞察

直接预测 $\mu_\theta(x_t, t)$ 比较困难，但**预测噪声**更容易：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

其中 $\epsilon_\theta(x_t, t)$ 是神经网络预测的噪声。

### 为什么预测噪声更容易？

1. **噪声是标准正态分布**：更容易学习
2. **目标简单**：只需要预测添加了多少噪声
3. **实践证明**：效果比直接预测均值更好

---

## 噪声预测网络

### U-Net 架构

扩散模型通常使用 **U-Net** 作为噪声预测网络：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        """
        U-Net 噪声预测网络

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
        """
        super().__init__()

        # 编码器（下采样）
        self.enc1 = self._block(in_channels, 64)
        self.enc2 = self._block(64, 128)
        self.enc3 = self._block(128, 256)

        # 瓶颈层
        self.bottleneck = self._block(256, 512)

        # 解码器（上采样）
        self.dec3 = self._block(512 + 256, 256)
        self.dec2 = self._block(256 + 128, 128)
        self.dec1 = self._block(128 + 64, 64)

        # 输出层
        self.out = nn.Conv2d(64, out_channels, 1)

    def _block(self, in_channels, out_channels):
        """基本卷积块"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, t):
        """
        前向传播

        输入: x (B, C, H, W), t (B,)
        输出: predicted_noise (B, C, H, W)
        """
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))

        # 瓶颈
        b = self.bottleneck(F.max_pool2d(e3, 2))

        # 解码（带跳跃连接）
        d3 = self.dec3(torch.cat([F.interpolate(b, scale_factor=2), e3], dim=1))
        d2 = self.dec2(torch.cat([F.interpolate(d3, scale_factor=2), e2], dim=1))
        d1 = self.dec1(torch.cat([F.interpolate(d2, scale_factor=2), e1], dim=1))

        return self.out(d1)
```

### 为什么用 U-Net？

1. **跳跃连接**：保留多尺度信息
2. **编码器-解码器结构**：压缩和恢复信息
3. **实践证明**：在图像生成任务中效果好

---

## 训练过程

### 训练目标

最小化预测噪声和真实噪声的 MSE：

$$L = E_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

### 训练流程

```python
def train_diffusion(model, dataloader, betas, epochs=100):
    """
    训练扩散模型

    1. 随机采样时间步 t
    2. 正向扩散得到 x_t
    3. 预测噪声
    4. 计算 MSE 损失
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for images in dataloader:
            # 1. 随机采样时间步
            t = torch.randint(0, len(betas), (images.shape[0],))

            # 2. 正向扩散
            x_t, noise = forward_diffusion(images, t, betas)

            # 3. 预测噪声
            predicted_noise = model(x_t, t)

            # 4. 计算损失
            loss = F.mse_loss(predicted_noise, noise)

            # 5. 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 采样过程

### 从噪声生成图像

```python
def sample(model, betas, shape):
    """
    从噪声生成图像

    参数:
        model: 训练好的噪声预测网络
        betas: 噪声调度
        shape: 生成图像的形状
    """
    # 1. 从纯噪声开始
    x = torch.randn(shape)

    # 2. 逐步去噪
    for t in reversed(range(len(betas))):
        # 预测噪声
        predicted_noise = model(x, torch.tensor([t]))

        # 计算参数
        alpha = 1 - betas[t]
        alpha_bar = torch.cumprod(alpha, dim=0)[t]

        # 去噪
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0

        # 更新 x
        x = (1 / torch.sqrt(alpha)) * (
            x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * predicted_noise
        ) + torch.sqrt(betas[t]) * noise

    return x
```

### 采样公式推导

从 $x_t$ 到 $x_{t-1}$：

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

其中 $z \sim N(0, I)$。

---

## DDPM 超参数

### 噪声调度

```python
# 线性噪声调度
num_timesteps = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, num_timesteps)
```

### 其他超参数

```python
config = {
    'num_timesteps': 1000,    # 时间步数
    'beta_start': 0.0001,     # 起始噪声
    'beta_end': 0.02,         # 结束噪声
    'batch_size': 32,         # 批次大小
    'lr': 1e-4,               # 学习率
    'epochs': 100,            # 训练轮数
}
```

---

## 条件生成

### 文本条件

用文本描述控制生成内容：

```python
class ConditionalUNet(UNet):
    def __init__(self, text_embed_dim=512):
        """
        条件 U-Net

        用文本嵌入作为条件
        """
        super().__init__()

        # 文本嵌入投影
        self.text_embed = nn.Linear(text_embed_dim, 512)

    def forward(self, x, t, text_embedding):
        """
        前向传播

        输入: x (B, C, H, W), t (B,), text_embedding (B, text_embed_dim)
        """
        # 投影文本嵌入
        text_cond = self.text_embed(text_embedding)

        # 将文本条件注入到网络中
        # （简化实现，实际会更复杂）
        ...
```

### 其他条件

- **类别条件**：用类别标签控制
- **图像条件**：用参考图像控制（如图像修复）
- **边缘条件**：用边缘图控制（如 ControlNet）

---

## 扩散模型 vs 其他生成模型

### 优势

| 特性 | GAN | VAE | 扩散模型 |
|------|-----|-----|----------|
| 训练稳定性 | 不稳定 | 稳定 | 稳定 |
| 生成质量 | 高 | 中 | 高 |
| 多样性 | 中 | 高 | 高 |
| 模式坍塌 | 有 | 无 | 无 |

### 劣势

| 特性 | GAN | VAE | 扩散模型 |
|------|-----|-----|----------|
| 生成速度 | 快 | 快 | 慢 |
| 计算开销 | 中 | 小 | 大 |
| 内存占用 | 中 | 小 | 大 |

---

## 深入讨论

### 为什么扩散模型生成质量高？

1. **逐步去噪**：每一步只做简单的去噪操作
2. **稳定的训练目标**：MSE 损失，不像 GAN 有对抗训练
3. **丰富的梯度信号**：每个时间步都有梯度

### 如何加速采样？

1. **DDIM**：跳过部分时间步
2. **DPM-Solver**：使用更高阶的求解器
3. **一致性模型**：直接从噪声映射到图像

### 扩散模型的应用

1. **图像生成**：Stable Diffusion、DALL-E 2
2. **图像编辑**：图像修复、风格迁移
3. **视频生成**：Video Diffusion
4. **音频生成**：Audio Diffusion

---

## 关键要点

1. **扩散模型**通过逐步添加和去除噪声来生成图像
2. **正向扩散**将图像变成噪声，**反向扩散**从噪声恢复图像
3. **U-Net** 是常用的噪声预测网络
4. **训练目标**是预测噪声，而不是直接预测图像
5. **条件生成**可以用文本、类别等控制生成内容

---

## 延伸阅读

1. **DDPM**：Denoising Diffusion Probabilistic Models
2. **DDIM**：Denoising Diffusion Implicit Models
3. **Stable Diffusion**：Latent Diffusion Models
4. **ControlNet**：条件控制的扩散模型
