# -*- coding: utf-8 -*-
# @Time : 2025/06/30
# @Author : lzm

"""
 @description  从 PDF 按页范围提取文本，供蒸馏笔记时对照原书

 @dependency  pymupdf (pip install pymupdf)

 @output       终端打印或写入 txt 文件
"""

import argparse
from pathlib import Path

import fitz


def extract_pages(pdf_path: str, start: int, end: int, out: str | None = None) -> str:
    """
    提取 PDF 指定页码范围的文本

    :param pdf_path: PDF 文件路径
    :param start: 起始页（1-based，含）
    :param end: 结束页（1-based，含）
    :param out: 可选输出文件路径
    :return: 拼接后的文本
    """
    doc = fitz.open(pdf_path)
    parts = []
    for i in range(start - 1, min(end, doc.page_count)):
        parts.append(f"\n{'='*60}\n第 {i+1} 页\n{'='*60}\n")
        parts.append(doc[i].get_text())
    text = "".join(parts)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(description="从 PDF 提取指定页文本")
    parser.add_argument("pdf", help="PDF 路径")
    parser.add_argument("--start", type=int, default=1, help="起始页（含）")
    parser.add_argument("--end", type=int, required=True, help="结束页（含）")
    parser.add_argument("-o", "--out", help="输出 txt 路径")
    args = parser.parse_args()
    text = extract_pages(args.pdf, args.start, args.end, args.out)
    if not args.out:
        print(text)


if __name__ == "__main__":
    main()
