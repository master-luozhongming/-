# 大模型 · 强化学习 · 多模态 — 学习笔记

> 来源：左元《大语言模型、强化学习和多模态教程》（407 页）  
> 人话版：[Modelshit Class](https://modelshit-class.vercel.app/)

## 第一部分 · 大语言模型

| 章 | 主题 | 笔记 |
|----|------|------|
| 01 | microGPT | [01_microgpt.md](llm/01_microgpt.md) |
| 02 | LLM 简介 | [02_LLM简介.md](llm/02_LLM简介.md) |
| 03 | GPT-2 | [03_GPT2.md](llm/03_GPT2.md) |

## 第二部分 · 强化学习

| 章 | 主题 | 笔记 |
|----|------|------|
| 04 | 强化学习简介 | [04_强化学习简介.md](rl/04_强化学习简介.md) |
| 05 | 策略梯度法 | [05_策略梯度法.md](rl/05_策略梯度法.md) |
| 06 | PPO | [06_PPO.md](rl/06_PPO.md) |
| 07 | PPO 实战 | [07_PPO实战.md](rl/07_PPO实战.md) |
| 08 | PPO 数学 | [08_PPO数学.md](rl/08_PPO数学.md) |
| 09 | GRPO | [09_GRPO.md](rl/09_GRPO.md) |

## 第三部分 · 对齐与微调

| 章 | 主题 | 笔记 |
|----|------|------|
| 10 | 训练全流程 | [10_训练全流程.md](align/10_训练全流程.md) |
| 11 | DPO | [11_DPO.md](align/11_DPO.md) |
| 12 | PPO / InstructGPT | [12_PPO_InstructGPT.md](align/12_PPO_InstructGPT.md) |
| 13 | GRPO / DeepSeek-R1 | [13_GRPO_DeepSeek.md](align/13_GRPO_DeepSeek.md) |

## 第四部分 · 多模态

| 章 | 主题 | 笔记 |
|----|------|------|
| 14 | Vision Transformer | [14_ViT.md](mm/14_ViT.md) |
| 15 | CLIP | [15_CLIP.md](mm/15_CLIP.md) |
| 16 | ClipCap | [16_ClipCap.md](mm/16_ClipCap.md) |
| 17 | 扩散模型 | [17_扩散模型.md](mm/17_扩散模型.md) |
| 18 | 扩散数学 | [18_扩散数学.md](mm/18_扩散数学.md) |
| 19 | 条件扩散 | [19_条件扩散.md](mm/19_条件扩散.md) |
| 20 | DALL-E 2 | [20_DALLE2.md](mm/20_DALLE2.md) |

## 附录

| 章 | 主题 | 笔记 |
|----|------|------|
| 21 | 全书总结 | [21_全书总结.md](appendix/21_全书总结.md) |
| A | uv 环境管理 | [A_uv环境管理.md](appendix/A_uv环境管理.md) |

## 工具

```bash
# 重新生成全部章节（需本地 PDF）
python tools/generate_all_docs.py

# 提取单章原文对照
python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 77 --end 110
```
