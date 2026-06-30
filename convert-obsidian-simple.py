#!/usr/bin/env python3
"""
直接复制 Obsidian 内容到 VitePress
"""

import os
import shutil
from pathlib import Path

# 源目录
source_dir = Path(r"E:\大模型\rl-tutorial-obsidian")
# 目标目录
target_dir = Path(r"D:\Users\master luo\PycharmProjects\reinforcement\docs")

# 章节映射 - 按照 modelshit-class.vercel.app 的内容结构
chapter_mapping = {
    # 强化学习基础
    "1. 强化学习基础.md": "02-rl-basics/01-concepts.md",

    # 策略梯度法
    "2. 策略梯度法.md": "03-policy-gradient/01-vanilla-pg.md",
    "附录C：广义优势估计（GAE）.md": "03-policy-gradient/02-reinforce.md",
    "附录E：策略梯度法的证明.md": "03-policy-gradient/03-baseline.md",
    "附录F：重要性采样.md": "03-policy-gradient/04-actor-critic.md",
    "附录G：uv教程.md": "03-policy-gradient/05-gae.md",

    # PPO
    "3. 策略梯度法的问题以及解决方案：PPO.md": "04-ppo/01-theory.md",
    "4. 使用 PPO 玩倒立摆游戏.md": "04-ppo/02-implementation.md",
    "附录D：PPO数学推导.md": "04-ppo/03-math.md",

    # RLHF
    "5. 基于人类反馈的强化学习（RLHF）.md": "05-rlhf/01-dpo.md",
    "6. 使用PPO微调LLM.md": "05-rlhf/02-grpo.md",
    "7. DPO：直接偏好优化.md": "05-rlhf/03-grpo-app.md",

    # 其他
    "8. 使用DPO微调LLM.md": "01-llm/01-microgpt.md",
    "9. 思维链（Chain of Thoughts）.md": "01-llm/02-llm-intro.md",
    "10. GRPO：组相对策略优化.md": "01-llm/03-gpt2.md",
    "11. 使用GRPO微调LLM.md": "06-multimodal/01-vit.md",
    "12. 强化学习面试题.md": "06-multimodal/02-clip.md",

    # 附录
    "附录A：自己实现倒立摆环境.md": "06-multimodal/03-diffusion.md",
    "附录B：KL散度.md": "02-rl-basics/02-value-function.md",
    "附录H：torch.gather.md": "02-rl-basics/03-bellman-equation.md",
}

def copy_images():
    """
    复制图片文件
    """
    source_img_dir = source_dir / "img"
    target_img_dir = target_dir / "public" / "img"

    if source_img_dir.exists():
        if target_img_dir.exists():
            shutil.rmtree(target_img_dir)
        shutil.copytree(source_img_dir, target_img_dir)
        print(f"复制图片: {source_img_dir} -> {target_img_dir}")

def copy_chapters():
    """
    复制章节内容
    """
    for source_name, target_name in chapter_mapping.items():
        source_file = source_dir / source_name
        target_file = target_dir / target_name

        if source_file.exists():
            # 确保目标目录存在
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # 复制文件
            shutil.copy2(source_file, target_file)

            print(f"复制章节: {source_name} -> {target_name}")

def main():
    """
    主函数
    """
    print("开始复制 Obsidian 内容到 VitePress...")

    # 1. 复制图片
    copy_images()

    # 2. 复制章节
    copy_chapters()

    print("复制完成！")

if __name__ == "__main__":
    main()
