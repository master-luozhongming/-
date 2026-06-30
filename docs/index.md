---
layout: home

hero:
  name: "强化学习与大模型教程"
  text: "理论与实践"
  tagline: 基于左元《大语言模型、强化学习和多模态教程》知识蒸馏
  image:
    src: /hero.png
    alt: RL Tutorial
  actions:
    - theme: brand
      text: 开始学习
      link: /02-rl-basics/01-concepts
    - theme: alt
      text: 查看代码
      link: https://github.com/master-luozhongming/-

features:
  - icon: 🤖
    title: 大语言模型
    details: 从 MicroGPT 到 GPT-2，理解 LLM 的核心原理和实现
    link: /01-llm/01-microgpt

  - icon: 🎮
    title: 强化学习基础
    details: 状态、动作、奖励、价值函数、贝尔曼方程
    link: /02-rl-basics/01-concepts

  - icon: 📈
    title: 策略梯度法
    details: REINFORCE、Actor-Critic、广义优势估计
    link: /03-policy-gradient/01-vanilla-pg

  - icon: 🚀
    title: PPO 算法
    details: 近端策略优化，最流行的策略梯度算法
    link: /04-ppo/01-theory

  - icon: 👥
    title: RLHF
    details: DPO、GRPO，基于人类反馈的强化学习
    link: /05-rlhf/01-dpo

  - icon: 🖼️
    title: 多模态
    details: Vision Transformer、CLIP、扩散模型
    link: /06-multimodal/01-vit
---

<style>
:root {
  --vp-home-hero-name-color: transparent;
  --vp-home-hero-name-background: -webkit-linear-gradient(120deg, #bd34fe 30%, #41d1ff);
}
</style>
