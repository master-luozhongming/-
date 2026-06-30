#!/usr/bin/env python3
"""
修复格式问题
"""

import os
import re

def fix_content(content):
    """
    修复内容格式
    """
    # 1. 删除文件开头的空列表项
    content = re.sub(r'^(\s*-\s*\n)+', '', content)

    # 2. 删除章节标题后面的 ¶ 符号
    content = content.replace('¶', '')

    # 3. 修复数学公式格式
    # \(...\) -> $...$
    content = re.sub(r'\\\((.*?)\\\)', r'$\1$', content)

    # 4. 修复块级数学公式
    # \[...\] -> $$...$$
    content = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', content, flags=re.DOTALL)

    # 5. 删除多余的空行
    content = re.sub(r'\n{3,}', '\n\n', content)

    # 6. 删除行尾的空格
    content = re.sub(r' +\n', '\n', content)

    # 7. 修复 mermaid 代码块
    # flowchart LR -> ```mermaid\nflowchart LR\n```
    content = re.sub(r'(flowchart\s+\w+)', r'```\n\1', content)
    content = re.sub(r'(sequenceDiagram)', r'```\n\1', content)

    return content

def main():
    print("开始修复格式问题...")

    # 需要修复的文件
    files_to_fix = [
        'docs/02-rl-basics/01-concepts.md',
        'docs/03-policy-gradient/01-vanilla-pg.md',
        'docs/04-ppo/01-theory.md',
        'docs/04-ppo/03-math.md',
    ]

    for file_path in files_to_fix:
        if os.path.exists(file_path):
            # 读取文件
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 修复内容
            fixed_content = fix_content(content)

            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)

            print(f"修复: {file_path}")
        else:
            print(f"跳过: {file_path} (文件不存在)")

    print("\n修复完成！")

if __name__ == '__main__':
    main()
