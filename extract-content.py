#!/usr/bin/env python3
"""
从 modelshit-class.vercel.app 提取内容
"""

import urllib.request
import re
from html.parser import HTMLParser

class ContentExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_article = False
        self.in_code = False
        self.content = []
        self.current_text = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == 'article':
            self.in_article = True
        elif tag == 'code':
            self.in_code = True
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
        elif tag == 'code':
            self.in_code = False
        elif tag in ['h1', 'h2', 'h3']:
            self.content.append('\n')
        elif tag == 'p':
            self.content.append('\n')

    def handle_data(self, data):
        if self.in_article:
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
        with open('modelshit-content.md', 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Content saved to modelshit-content.md")
        print(f"Content length: {len(content)} characters")

        # 打印前 500 个字符
        print("\nFirst 500 characters:")
        print(content[:500])

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
