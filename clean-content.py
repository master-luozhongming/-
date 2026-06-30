#!/usr/bin/env python3
"""
清理和转换内容格式
"""

import re

def clean_content(content):
    """
    清理内容格式
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

    # 7. 删除开头的空行
    content = content.lstrip('\n')

    return content

def main():
    print("开始清理内容...")

    # 读取文件
    with open('modelshit-v2.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # 清理内容
    cleaned_content = clean_content(content)

    # 写入文件
    with open('docs/04-ppo/03-math.md', 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

    print("清理完成！")

if __name__ == '__main__':
    main()
