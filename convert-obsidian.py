#!/usr/bin/env python3
"""
将 Obsidian 格式的内容转换为 VitePress 格式
"""

import os
import re
import shutil
from pathlib import Path

# 源目录
source_dir = Path(r"E:\大模型\rl-tutorial-obsidian")
# 目标目录
target_dir = Path(r"D:\Users\master luo\PycharmProjects\reinforcement\docs")

# 章节映射 - 按照 Obsidian 文件的实际内容顺序
chapter_mapping = {
    "1. 强化学习基础.md": "02-rl-basics/01-concepts.md",
    "2. 策略梯度法.md": "03-policy-gradient/01-vanilla-pg.md",
    "3. 策略梯度法的问题以及解决方案：PPO.md": "04-ppo/01-theory.md",
    "4. 使用 PPO 玩倒立摆游戏.md": "04-ppo/02-implementation.md",
    "5. 基于人类反馈的强化学习（RLHF）.md": "05-rlhf/01-dpo.md",
    "6. 使用PPO微调LLM.md": "05-rlhf/02-grpo.md",
    "7. DPO：直接偏好优化.md": "05-rlhf/03-grpo-app.md",
    "8. 使用DPO微调LLM.md": "06-multimodal/01-vit.md",
    "9. 思维链（Chain of Thoughts）.md": "06-multimodal/02-clip.md",
    "10. GRPO：组相对策略优化.md": "06-multimodal/03-diffusion.md",
    "11. 使用GRPO微调LLM.md": "01-llm/01-microgpt.md",
    "12. 强化学习面试题.md": "01-llm/02-llm-intro.md",
}

# 附录映射
appendix_mapping = {
    "附录A：自己实现倒立摆环境.md": "02-rl-basics/02-value-function.md",
    "附录B：KL散度.md": "02-rl-basics/03-bellman-equation.md",
    "附录C：广义优势估计（GAE）.md": "03-policy-gradient/02-reinforce.md",
    "附录D：PPO数学推导.md": "04-ppo/03-math.md",
    "附录E：策略梯度法的证明.md": "03-policy-gradient/03-baseline.md",
    "附录F：重要性采样.md": "03-policy-gradient/04-actor-critic.md",
    "附录G：uv教程.md": "03-policy-gradient/05-gae.md",
    "附录H：torch.gather.md": "01-llm/03-gpt2.md",
}

def convert_obsidian_to_vitepress(content, source_file):
    """
    将 Obsidian 格式转换为 VitePress 格式
    """
    # 1. 转换图片链接
    # ![[image.excalidraw|1000]] -> ![image](/img/image.png)
    def convert_image_link(match):
        image_name = match.group(1)
        size = match.group(2) if match.group(2) else ""

        # 检查是否是 excalidraw 文件
        if "excalidraw" in image_name:
            # excalidraw 文件需要转换为图片
            # 这里假设已经转换为 png 格式
            png_name = image_name.replace(".excalidraw", ".png")
            if size:
                return f"![{png_name}](/img/{png_name}){{ width={size} }}"
            else:
                return f"![{png_name}](/img/{png_name})"
        else:
            # 普通图片
            if size:
                return f"![{image_name}](/img/{image_name}){{ width={size} }}"
            else:
                return f"![{image_name}](/img/{image_name})"

    content = re.sub(r'!\[\[([^|\]]+)(?:\|([^\]]+))?\]\]', convert_image_link, content)

    # 2. 转换特殊语法
    # > [!NOTE] -> ::: info
    content = re.sub(r'> \[!NOTE\]', '::: info', content)
    content = re.sub(r'> \[!TIP\]', '::: tip', content)
    content = re.sub(r'> \[!WARNING\]', '::: warning', content)
    content = re.sub(r'> \[!DANGER\]', '::: danger', content)

    # 3. 转换代码块
    # ```ad-note -> ::: info
    content = re.sub(r'```ad-note\n(.*?)```', r'::: info\n\1\n:::', content, flags=re.DOTALL)
    content = re.sub(r'```ad-tip\n(.*?)```', r'::: tip\n\1\n:::', content, flags=re.DOTALL)
    content = re.sub(r'```ad-danger\n(.*?)```', r'::: danger\n\1\n:::', content, flags=re.DOTALL)

    # 4. 转换数学公式
    # 保持原样，VitePress 支持 LaTeX

    # 5. 转换内部链接
    # [[link]] -> [link](/path/to/link)
    def convert_internal_link(match):
        link_name = match.group(1)
        # 这里简化处理，实际需要根据链接名称找到对应的页面
        return f"[{link_name}](/{link_name})"

    content = re.sub(r'\[\[([^\]]+)\]\]', convert_internal_link, content)

    return content

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

def convert_chapters():
    """
    转换章节内容
    """
    # 转换主章节
    for source_name, target_name in chapter_mapping.items():
        source_file = source_dir / source_name
        target_file = target_dir / target_name

        if source_file.exists():
            # 读取源文件
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 转换内容
            converted_content = convert_obsidian_to_vitepress(content, source_file)

            # 确保目标目录存在
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # 写入目标文件
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(converted_content)

            print(f"转换章节: {source_name} -> {target_name}")

    # 转换附录
    for source_name, target_name in appendix_mapping.items():
        source_file = source_dir / source_name
        target_file = target_dir / target_name

        if source_file.exists():
            # 读取源文件
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 转换内容
            converted_content = convert_obsidian_to_vitepress(content, source_file)

            # 确保目标目录存在
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # 写入目标文件
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(converted_content)

            print(f"转换附录: {source_name} -> {target_name}")

def main():
    """
    主函数
    """
    print("开始转换 Obsidian 内容到 VitePress...")

    # 1. 复制图片
    copy_images()

    # 2. 转换章节
    convert_chapters()

    print("转换完成！")

if __name__ == "__main__":
    main()
