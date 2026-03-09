# 🍼 BabyGrok: 从零手搓混合专家 (MoE) 大语言模型

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

本项目是一个极简、纯 PyTorch 实现的 **大语言模型 (LLM)**。它的架构灵感来源于 xAI 的 **Grok-1**，核心特色是包含了 **MoE (Mixture-of-Experts, 混合专家)** 机制。

本项目专为**初学者**和**只有 Python 基础**的开发者设计，去除了冗杂的工程化代码，保留了最原汁原味的 Transformer 和 MoE 核心逻辑。即使你只有一张 **4GB 显存**的入门级显卡，也能在本地顺利训练出一个会讲故事的小模型！

---

## ✨ 核心特性

- **🧠 纯正 Transformer 架构**：包含 Causal Self-Attention (因果自注意力) 和残差连接。
- **⚡ 混合专家机制 (MoE)**：实现了包含 Router (路由器) 和多个 Experts (专家) 的动态分发层，告别臃肿的稠密计算。
- **💾 极致显存优化**：专为 4GB VRAM (如 GTX 1650/3050) 优化，内置混合精度训练 (AMP)、梯度累积和极速 Flash Attention。
- **📖 零基础友好**：代码结构清晰，注释详尽，将“深度学习”还原为基础的 Python 面向对象编程。

---

## 📂 项目结构

```text
BabyGrok-Project/
│
├── config.py             # ⚙️ 核心配置文件 (模型大小、学习率、Batch Size 等)
├── model.py              # 🧠 模型结构定义 (Attention, MoE, Block 等核心逻辑)
├── prepare_data.py       # 🗂️ 数据下载与处理 (使用 tiktoken 将文本转为数字特征)
├── train.py              # 🚀 训练脚本 (包含前向传播、Loss 计算、反向传播与保存机制)
├── chat.py               # 💬 交互式推理脚本 (加载训练好的模型并进行对话)
│
├── data/                 # 存放处理好的二进制训练数据 (如 train.bin)
├── out/                  # 存放训练输出的模型权重 (如 model_latest.pth)
└── README.md             # 项目说明文档
