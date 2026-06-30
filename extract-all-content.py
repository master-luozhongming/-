#!/usr/bin/env python3
"""
从 modelshit-class.vercel.app 提取所有章节内容
"""

import urllib.request
import re
from html.parser import HTMLParser

class ContentExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_article = False
        self.content = []

    def handle_starttag(self, tag, attrs):
        if tag == 'article':
            self.in_article = True
        elif tag == 'h1':
            self.content.append('\n# ')
        elif tag == 'h2':
            self.content.append('\n## ')
        elif tag == 'h3':
            self.content.append('\n### ')
        elif tag == 'p':
            self.content.append('\n')
        elif tag == 'li':
            self.content.append('\n- ')

    def handle_endtag(self, tag):
        if tag == 'article':
            self.in_article = False
        elif tag in ['h1', 'h2', 'h3']:
            self.content.append('\n')
        elif tag == 'p':
            self.content.append('\n')

    def handle_data(self, data):
        if self.in_article:
            self.content.append(data)

    def get_content(self):
        return ''.join(self.content)

def fetch_page(url):
    """获取页面内容"""
    try:
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')

        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8')

        parser = ContentExtractor()
        parser.feed(html)

        return parser.get_content()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None

def main():
    base_url = 'https://modelshit-class.vercel.app/01-rl/'

    # 章节列表
    chapters = [
        ('04-rl-intro/', '01-rl-intro.md', '强化学习简介'),
        ('05-policy-gradient/', '02-policy-gradient.md', '策略梯度法'),
        ('06-ppo/', '03-ppo.md', 'PPO'),
        ('07-ppo-cartpole/', '04-ppo-cartpole.md', 'PPO 倒立摆'),
        ('08-ppo-math/', '05-ppo-math.md', 'PPO 数学推导'),
        ('09-rlhf/', '06-rlhf.md', 'RLHF'),
        ('10-ppo-llm/', '07-ppo-llm.md', 'PPO 微调 LLM'),
        ('11-dpo/', '08-dpo.md', 'DPO'),
        ('12-dpo-llm/', '09-dpo-llm.md', 'DPO 微调 LLM'),
        ('13-cot/', '10-cot.md', '思维链'),
        ('14-grpo/', '11-grpo.md', 'GRPO'),
        ('15-grpo-llm/', '12-grpo-llm.md', 'GRPO 微调 LLM'),
    ]

    print("开始提取所有章节内容...")

    for chapter_url, filename, title in chapters:
        url = base_url + chapter_url
        print(f"提取: {title} ({url})")

        content = fetch_page(url)

        if content:
            # 保存到文件
            with open(f'modelshit-chapters/{filename}', 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  -> 已保存到 modelshit-chapters/{filename}")
        else:
            print(f"  -> 提取失败")

    print("\n提取完成！")

if __name__ == '__main__':
    import os
    os.makedirs('modelshit-chapters', exist_ok=True)
    main()
