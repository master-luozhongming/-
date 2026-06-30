#!/bin/bash

# Git 初始化和推送脚本

echo "=========================================="
echo "强化学习知识蒸馏项目 - Git 初始化脚本"
echo "=========================================="

# 检查是否在正确的目录
if [ ! -f "README.md" ]; then
    echo "错误：请在项目根目录运行此脚本"
    exit 1
fi

# 初始化 Git 仓库
echo ""
echo "1. 初始化 Git 仓库..."
git init

# 添加所有文件
echo ""
echo "2. 添加所有文件..."
git add .

# 首次提交
echo ""
echo "3. 首次提交..."
git commit -m "feat: 初始化强化学习知识蒸馏项目

- 添加强化学习基础概念文档
- 添加策略梯度法（REINFORCE）详解
- 添加 PPO 理论和实现文档
- 添加 GRPO 和 DPO 文档
- 添加完整代码示例
- 添加使用说明和 GitHub 推送指南"

echo ""
echo "=========================================="
echo "Git 仓库初始化完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 在 GitHub 上创建新仓库"
echo "2. 运行以下命令关联远程仓库："
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/reinforcement-learning-tutorial.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "详细说明请查看 GITHUB.md 文件"
echo ""
