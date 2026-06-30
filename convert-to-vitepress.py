#!/usr/bin/env python3
"""
将提取的 modelshit 内容转换为 VitePress 格式
"""

import os
import re

# 章节映射
chapter_mapping = {
    'modelshit-chapters/01-rl-intro.md': 'docs/02-rl-basics/01-concepts.md',
    'modelshit-chapters/02-policy-gradient.md': 'docs/03-policy-gradient/01-vanilla-pg.md',
    'modelshit-chapters/03-ppo.md': 'docs/04-ppo/01-theory.md',
    'modelshit-chapters/05-ppo-math.md': 'docs/04-ppo/03-math.md',
}

def convert_obsidian_to_vitepress(content):
    """
    将 Obsidian 格式转换为 VitePress 格式
    """
    # 1. 转换图片链接
    # ![image](url) -> ![image](url) (保持原样)

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

    # 4. 转换 details 标签
    # 想看... -> ::: details 想看...
    content = re.sub(r'想看([^：\n]+)（可跳过）', r'::: details 想看\1（可跳过）', content)

    # 5. 转换数学公式
    # \( -> $, \) -> $
    content = re.sub(r'\\\((.*?)\\\)', r'$\1$', content)

    # 6. 转换表格
    # 保持原样，VitePress 支持 Markdown 表格

    return content

def main():
    print("开始转换内容为 VitePress 格式...")

    for source_file, target_file in chapter_mapping.items():
        if os.path.exists(source_file):
            # 读取源文件
            with open(source_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 转换内容
            converted_content = convert_obsidian_to_vitepress(content)

            # 确保目标目录存在
            os.makedirs(os.path.dirname(target_file), exist_ok=True)

            # 写入目标文件
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(converted_content)

            print(f"转换: {source_file} -> {target_file}")
        else:
            print(f"跳过: {source_file} (文件不存在)")

    print("\n转换完成！")

if __name__ == '__main__':
    main()
