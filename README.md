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
```

## 🚀 快速开始 (保姆级教程)

### 1. 环境准备

确保你的电脑安装了 Python，并建议配备一张 NVIDIA 显卡 (需安装 CUDA)。
打开终端，安装所需的依赖库：

```
pip install torch numpy tiktoken datasets
```

### 2. 准备“饲料” (数据处理)

模型需要阅读大量的文本才能学会说话。运行以下脚本，它会自动下载测试数据集（如 TinyStories）并将其编码为模型认识的二进制格式 train.bin。

```
python prepare_data.py
```

*(注：处理完成后，data/ 目录下会生成一个 .bin 文件。)*

### 3. 开始炼丹 (模型训练)

启动训练脚本。脚本会自动检测你是否有 GPU。

```
python train.py
```

**训练提示：**

- 刚开始的 Step 0 可能会卡顿 10-20 秒，这是显卡在进行 CUDA 预热，属正常现象。
- 观察终端输出的 Loss 值，如果它从 10.x 逐渐下降到 3.0 以下，说明模型正在变聪明！
- 训练过程中如果想提前结束，可以直接按 Ctrl+C，代码会自动保存当前的进度为 model_interrupted.pth。

### 4. 见证奇迹 (与模型对话)

当训练完成（或你在中途保存了模型权重），运行聊天脚本：

```
python chat.py
```

在终端中输入一段英文开头（例如 "Once upon a time"），按下回车，看着你的 BabyGrok 逐字续写故事吧！

------



## 🛠️ 进阶配置与显存调优 (OOM 救星)

如果你在运行 train.py 时遇到了 CUDA Out of Memory (爆显存) 错误，或者觉得训练太慢，请打开 config.py 进行调整：

- **如果爆显存 (OOM)**：调小 batch_size（如 64 -> 32 -> 16），或者减小 dim（如 256 -> 128）。
- **如果显卡占用率低 (GPU 摸鱼)**：调大 batch_size，让数据填满你的显存（建议显存占用保持在 85% 左右最佳）。
- **如果想让模型更聪明**：在显存允许的情况下，增加 n_layers (层数) 和 dim (维度)，并准备更多的数据进行更长时间的训练。

------



## 🎓 学习路线指南

如果你想看懂 model.py 里的代码，建议按照以下顺序阅读：

1. 
2. **Expert 类**：最基础的全连接神经网络。
3. **MoELayer 类**：理解 Router 是如何打分，并通过 Mask (掩码) 将数据分发给特定专家的。
4. **CausalSelfAttention 类**：理解 Q、K、V 是如何计算相关性，以及下三角矩阵是如何防止模型“偷看未来”的。
5. **Block 与 BabyGrok 类**：看懂积木是如何一层层拼装起来的。

------



## 🙏 致谢

- 感谢 **[xAI Grok-1](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Fxai-org%2Fgrok-1)** 提供的 MoE 架构灵感。
- 感谢 **[Andrej Karpathy](https://www.google.com/url?sa=E&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2FnanoGPT)** 的教学项目带来的启发，本项目致力于用同样大道至简的理念降低大模型学习门槛。
