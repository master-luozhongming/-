# -*- coding: utf-8 -*-
# @Time : 2025/06/30
# @Author : lzm

"""
 @description  从 PDF 批量生成全书蒸馏 Markdown

 @dependency  pymupdf

 @output       docs/ 下各章 markdown
"""

import re
import sys
from pathlib import Path

import fitz

PDF = Path(r"E:\大模型\强化学习\main2.pdf")
DOCS = Path(__file__).resolve().parent.parent / "docs"
BASE_URL = "https://modelshit-class.vercel.app"

CHAPTERS = [
    # (part_dir, filename, title, pdf_start, pdf_end, web_path, intro)
    ("llm", "01_microgpt", "microGPT：60 行看懂大模型全貌", 17, 32,
     "/01-llm/01-microgpt/",
     "用 4192 参数的迷你 GPT 串起 embedding、Transformer、训练与推理全流程。"),
    ("llm", "02_LLM简介", "大模型在干嘛：预测下一个词", 33, 36,
     "/01-llm/02-next-token/",
     "LLM 本质是条件概率 P(下一 token | 上文)，训练数据是输入-目标对。"),
    ("llm", "03_GPT2", "GPT-2：把 Transformer 拆开看", 37, 62,
     "/01-llm/03-gpt2/",
     "Decoder-only Transformer：因果注意力、LayerNorm、FFN、位置编码与训练循环。"),
    ("rl", "04_强化学习简介", "强化学习是什么", 63, 76,
     "/01-rl/04-what-is-rl/",
     "MDP 五元组、策略/价值函数、CartPole 环境、贝尔曼方程。"),
    ("rl", "05_策略梯度法", "策略梯度法", 77, 110,
     "/01-rl/05-policy-gradient/",
     "母公式不变，只改权重 Φ：原始PG → REINFORCE → 基线 → Actor-Critic → GAE。"),
    ("rl", "06_PPO", "PPO：稳一点的策略更新", 111, 118,
     "/01-rl/06-ppo/",
     "clip 限制策略更新幅度，防止一次异常奖励把策略拽崩。"),
    ("rl", "07_PPO实战", "PPO 实战", 119, 124,
     "/01-rl/07-ppo-practice/",
     "Actor-Critic 网络、GAE 计算、PPO 训练循环在 CartPole 上的完整实现。"),
    ("rl", "08_PPO数学", "PPO 背后的数学", 125, 140,
     "/01-rl/08-ppo-math/",
     "KL 散度、重要性采样、替代目标函数与误差上界。"),
    ("rl", "09_GRPO", "GRPO：去掉评论家", 141, 147,
     "/01-rl/09-grpo/",
     "组内相对奖励替代 Critic，DeepSeek-R1 同款思路的简化版。"),
    ("align", "10_训练全流程", "大模型训练全流程", 149, 154,
     "/02-align/10-llm-training/",
     "预训练 → SFT → RLHF/DPO/GRPO，对齐技术选型一览。"),
    ("align", "11_DPO", "DPO：不用强化学习也能对齐", 155, 180,
     "/02-align/11-dpo/",
     "把偏好学习改写成分类损失，省掉奖励模型和 PPO 采样循环。"),
    ("align", "12_PPO_InstructGPT", "用 PPO 复刻 InstructGPT", 181, 202,
     "/02-align/12-ppo-instructgpt/",
     "SFT → 奖励模型 → PPO 三阶段，RLHF 工业标准管线。"),
    ("align", "13_GRPO_DeepSeek", "用 GRPO 复刻 DeepSeek-R1", 203, 239,
     "/02-align/13-grpo-deepseek/",
     "DAPO/GRPO 微调 Qwen，奖励函数设计与推理链场景。"),
    ("mm", "14_ViT", "Vision Transformer", 241, 254,
     "/03-mm/14-vit/",
     "图像切 Patch 当 token，用 Transformer 做图像分类。"),
    ("mm", "15_CLIP", "CLIP：让图文站到同一空间", 255, 272,
     "/03-mm/15-clip/",
     "双塔编码器 + 对比学习，图文共享嵌入空间，支持零样本分类。"),
    ("mm", "16_ClipCap", "ClipCap：看图说话", 273, 286,
     "/03-mm/16-clipcap/",
     "CLIP 视觉特征 + GPT 文本解码，实现图像描述生成。"),
    ("mm", "17_扩散模型", "扩散模型：从噪声里画画", 287, 318,
     "/03-mm/17-diffusion/",
     "正向加噪、反向去噪，U-Net 预测噪声，DDPM 训练与采样。"),
    ("mm", "18_扩散数学", "扩散模型背后的数学", 319, 332,
     "/03-mm/18-diffusion-math/",
     "变分下界、重参数化、得分匹配与 ELBO 推导。"),
    ("mm", "19_条件扩散", "条件扩散：让它画我想要的", 333, 358,
     "/03-mm/19-conditional-diffusion/",
     "Classifier Guidance / Classifier-Free Guidance，条件控制生成。"),
    ("mm", "20_DALLE2", "DALL-E 2：文生图全家桶", 359, 400,
     "/03-mm/20-dalle2/",
     "CLIP 语义空间 + 先验 + 解码器，文本到图像的级联 pipeline。"),
    ("appendix", "21_全书总结", "全书总结", 401, 402,
     "/appendix/21-summary/",
     "四大部分知识地图与推荐阅读顺序。"),
    ("appendix", "A_uv环境管理", "uv：Python 环境管理", 403, 403,
     "/appendix/a-uv/",
     "用 uv 管理 Python 依赖与虚拟环境，替代 pip+venv。"),
]


def clean_text(text: str) -> str:
    """清洗 PDF 提取文本"""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def extract_section_titles(text: str) -> list[str]:
    """提取 x.y 形式小节标题"""
    titles = []
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r"^\d+(\.\d+)*\s+\S", line) and len(line) < 80:
            if not re.match(r"^\d+\s*$", line):
                titles.append(line)
    seen = set()
    out = []
    for t in titles:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:25]


def extract_key_paragraphs(text: str, max_n: int = 8) -> list[str]:
    """提取有实质内容的段落"""
    paras = []
    for p in text.split("\n\n"):
        p = p.strip()
        p = re.sub(r"\s+", " ", p)
        if len(p) < 40 or re.match(r"^[\d\.]+$", p):
            continue
        if "Chapter" in p and len(p) < 30:
            continue
        if p not in paras:
            paras.append(p)
        if len(paras) >= max_n:
            break
    return paras


def build_chapter(part: str, fname: str, title: str, start: int, end: int,
                  web: str, intro: str, doc: fitz.Document) -> str:
    """生成单章 Markdown"""
    raw = ""
    for p in range(start - 1, min(end, doc.page_count)):
        raw += doc[p].get_text() + "\n"
    raw = clean_text(raw)
    sections = extract_section_titles(raw)
    paras = extract_key_paragraphs(raw)

    lines = [
        f"# {title.split('：')[0] if '：' in title else title}",
        "",
        f"> 原书 PDF 第 {start}–{end} 页",
        f"> 人话版：[Modelshit Class · {title}]({BASE_URL}{web})",
        "",
        "## 1. 一句话",
        "",
        intro,
        "",
        "## 2. 章节结构",
        "",
    ]
    if sections:
        for s in sections:
            lines.append(f"- {s}")
    else:
        lines.append("- （见原书目录）")

    lines += ["", "## 3. 核心要点", ""]
    if paras:
        for i, p in enumerate(paras, 1):
            if len(p) > 500:
                p = p[:497] + "..."
            lines.append(f"{i}. {p}")
            lines.append("")
    else:
        lines.append("详见原书对应章节与人话版链接。")
        lines.append("")

    lines += [
        "## 4. 与前后章关系",
        "",
        f"- 在线阅读：{BASE_URL}{web}",
        "- 本地对照：`python tools/extract_pdf.py \"E:\\大模型\\强化学习\\main2.pdf\" "
        f"--start {start} --end {end}`",
        "",
    ]
    return "\n".join(lines)


def main():
    if not PDF.exists():
        print(f"PDF 不存在: {PDF}", file=sys.stderr)
        sys.exit(1)
    doc = fitz.open(str(PDF))
    for part, fname, title, start, end, web, intro in CHAPTERS:
        out_dir = DOCS / part
        out_dir.mkdir(parents=True, exist_ok=True)
        md = build_chapter(part, fname, title, start, end, web, intro, doc)
        path = out_dir / f"{fname}.md"
        path.write_text(md, encoding="utf-8")
        print(f"OK {path.relative_to(DOCS.parent)}")
    doc.close()


if __name__ == "__main__":
    main()
