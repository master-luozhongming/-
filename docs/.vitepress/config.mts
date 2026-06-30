import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "强化学习与大模型教程",
  description: "理论与实践 - 知识蒸馏",
  lang: 'zh-CN',
  base: '/-/',

  // 忽略其他有问题的目录
  srcExclude: ['**/align/**', '**/mm/**', '**/tools/**'],

  // 禁用死链接检查
  ignoreDeadLinks: true,

  themeConfig: {
    logo: '/logo.svg',
    siteTitle: 'RL Tutorial',

    nav: [
      { text: '首页', link: '/' },
      { text: '开始学习', link: '/02-rl-basics/01-concepts' },
      {
        text: '章节',
        items: [
          { text: '大语言模型', link: '/01-llm/01-microgpt' },
          { text: '强化学习基础', link: '/02-rl-basics/01-concepts' },
          { text: '策略梯度法', link: '/03-policy-gradient/01-vanilla-pg' },
          { text: 'PPO', link: '/04-ppo/01-theory' },
          { text: 'RLHF', link: '/05-rlhf/01-dpo' },
          { text: '多模态', link: '/06-multimodal/01-vit' }
        ]
      },
      { text: '代码', link: 'https://github.com/master-luozhongming/-/tree/main/code' }
    ],

    sidebar: {
      '/01-llm/': [
        {
          text: '大语言模型',
          items: [
            { text: '1. MicroGPT', link: '/01-llm/01-microgpt' },
            { text: '2. LLM 简介', link: '/01-llm/02-llm-intro' },
            { text: '3. GPT-2', link: '/01-llm/03-gpt2' }
          ]
        }
      ],
      '/02-rl-basics/': [
        {
          text: '强化学习基础',
          items: [
            { text: '1. 基本概念', link: '/02-rl-basics/01-concepts' },
            { text: '2. 价值函数', link: '/02-rl-basics/02-value-function' },
            { text: '3. 贝尔曼方程', link: '/02-rl-basics/03-bellman-equation' }
          ]
        }
      ],
      '/03-policy-gradient/': [
        {
          text: '策略梯度法',
          items: [
            { text: '1. 原始策略梯度法', link: '/03-policy-gradient/01-vanilla-pg' },
            { text: '2. REINFORCE', link: '/03-policy-gradient/02-reinforce' },
            { text: '3. 带基线的策略梯度', link: '/03-policy-gradient/03-baseline' },
            { text: '4. Actor-Critic', link: '/03-policy-gradient/04-actor-critic' },
            { text: '5. 广义优势估计', link: '/03-policy-gradient/05-gae' }
          ]
        }
      ],
      '/04-ppo/': [
        {
          text: '近端策略优化 (PPO)',
          items: [
            { text: '1. PPO 理论', link: '/04-ppo/01-theory' },
            { text: '2. PPO 实现', link: '/04-ppo/02-implementation' },
            { text: '3. PPO 数学推导', link: '/04-ppo/03-math' }
          ]
        }
      ],
      '/05-rlhf/': [
        {
          text: '基于人类反馈的强化学习',
          items: [
            { text: '1. DPO', link: '/05-rlhf/01-dpo' },
            { text: '2. GRPO', link: '/05-rlhf/02-grpo' },
            { text: '3. GRPO 应用', link: '/05-rlhf/03-grpo-app' }
          ]
        }
      ],
      '/06-multimodal/': [
        {
          text: '多模态',
          items: [
            { text: '1. Vision Transformer', link: '/06-multimodal/01-vit' },
            { text: '2. CLIP', link: '/06-multimodal/02-clip' },
            { text: '3. 扩散模型', link: '/06-multimodal/03-diffusion' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/master-luozhongming/-' }
    ],

    footer: {
      message: '基于左元《大语言模型、强化学习和多模态教程》整理',
      copyright: '© 2024 强化学习教程'
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索' },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: { selectText: '选择', navigateText: '切换', closeText: '关闭' }
          }
        }
      }
    },

    outline: {
      label: '页面导航',
      level: [2, 3]
    },

    docFooter: {
      prev: '上一页',
      next: '下一页'
    }
  }
})
