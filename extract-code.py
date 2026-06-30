#!/usr/bin/env python3
"""
提取代码块内容
"""

import urllib.request
import re

def main():
    url = 'https://modelshit-class.vercel.app/01-rl/05-policy-gradient/'

    try:
        # 发送请求
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0')

        with urllib.request.urlopen(req, timeout=30) as response:
            html = response.read().decode('utf-8')

        # 提取代码块
        # <pre><code>...</code></pre>
        code_blocks = re.findall(r'<pre><code>(.*?)</code></pre>', html, re.DOTALL)

        print(f"Found {len(code_blocks)} code blocks")

        for i, code in enumerate(code_blocks):
            # 清理 HTML 标签
            code = re.sub(r'<[^>]+>', '', code)
            print(f"\n--- Code Block {i+1} ---")
            print(code[:500])

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
