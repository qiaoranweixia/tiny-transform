from dataclasses import dataclass
import torch


@dataclass
class ModelConfig:
    # 模型架构参数
    dim: int = 256  # 嵌入维度 (由小开始，显卡好可改为 512/768)
    n_layers: int = 6  # Transformer 层数
    n_heads: int = 8  # 注意力头数
    n_kv_heads: int = 8
    vocab_size: int = 50304  # GPT-2 词表大小 (凑整方便计算)
    max_seq_len: int = 256  # 上下文窗口长度

    # MoE 参数
    num_experts: int = 4  # 专家总数
    num_experts_per_tok: int = 2  # 每个 token 激活几个专家

    # 训练参数
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 5000  # 训练步数
    eval_interval: int = 200  # 每多少步打印一次 loss
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'