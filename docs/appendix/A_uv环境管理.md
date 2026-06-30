# uv

> 原书 PDF 第 403–403 页
> 人话版：[Modelshit Class · uv：Python 环境管理](https://modelshit-class.vercel.app/appendix/a-uv/)

## 1. 一句话

用 uv 管理 Python 依赖与虚拟环境，替代 pip+venv。

## 2. 章节结构

- （见原书目录）

## 3. 核心要点

1. A. uv 教程 1 $ pip install uv Shell 2 $ uv venv rlhf-env # 创建虚拟环境 3 $ source rlhf-env/bin/activate # 进入虚拟环境 4 $ uv pip install jinja2 5 $ uv pip install pandas 6 $ uv pip install pyarrow 7 $ uv pip install pyyaml 8 $ uv pip install safetensors 9 $ uv pip install tensorboard 10 $ uv pip install tokenizers 11 $ uv pip install torch 12 $ uv pip install transformers 配置国内源 1 # 推荐使用清华源 Shell 2 $ echo 'export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"'>> ~/.bashrc 3 4 # 让配置立即生效 5 $ so...

## 4. 与前后章关系

- 在线阅读：https://modelshit-class.vercel.app/appendix/a-uv/
- 本地对照：`python tools/extract_pdf.py "E:\大模型\强化学习\main2.pdf" --start 403 --end 403`
