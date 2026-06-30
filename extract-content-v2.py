#!/usr/bin/env python3
"""
从 modelshit-class.vercel.app 提取内容，保持原始格式
"""

import urllib.request
import re
from html.parser import HTMLParser

class ContentExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_article = False
        self.in_code = False
        self.in_pre = False
        self.in_details = False
        self.content = []
        self.current_tag = None

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag == 'article':
            self.in_article = True
        elif tag == 'pre':
            self.in_pre = True
            self.content.append('\n```\n')
        elif tag == 'code':
            self.in_code = True
        elif tag == 'details':
            self.in_details = True
            self.content.append('\n::: details ')
        elif tag == 'summary':
            pass  # summary 内容会在 handle_data 中处理
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
        elif tag == 'br':
            self.content.append('\n')

        self.current_tag = tag

    def handle_endtag(self, tag):
        if tag == 'article':
            self.in_article = False
        elif tag == 'pre':
            self.in_pre = False
            self.content.append('\n```\n')
        elif tag == 'code':
            self.in_code = False
        elif tag == 'details':
            self.in_details = False
            self.content.append('\n:::\n')
        elif tag in ['h1', 'h2', 'h3']:
            self.content.append('\n')
        elif tag == 'p':
            self.content.append('\n')

        self.current_tag = None

    def handle_data(self, data):
        if self.in_article:
            # 处理 summary 标签的内容
            if self.current_tag == 'summary':
                self.content.append(data)
            elif not self.in_pre:
                self.content.append(data)

    def get_content(self):
        return ''.join(self.content)

def main():
    url = 'https://modelshit-class.vercel.app/01-rl/05-policy-gradient/'

    try:
        # 发送请求
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')

        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8')

        # 解析 HTML
        parser = ContentExtractor()
        parser.feed(html)

        # 获取内容
        content = parser.get_content()

        # 保存到文件
        with open('modelshit-v2.md', 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Content saved to modelshit-v2.md")
        print(f"Content length: {len(content)} characters")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
