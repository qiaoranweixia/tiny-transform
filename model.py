import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class Expert(nn.Module):
    """ 单个专家网络 (MLP) """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden_dim = 4 * cfg.dim
        self.net = nn.Sequential(
            nn.Linear(cfg.dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cfg.dim)
        )

    def forward(self, x):
        return self.net(x)


class MoELayer(nn.Module):
    """ 混合专家层 """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.k = cfg.num_experts_per_tok
        self.router = nn.Linear(cfg.dim, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(cfg) for _ in range(cfg.num_experts)])

    def forward(self, x):
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)

        # 路由计算
        logits = self.router(x_flat)
        weights, indices = torch.topk(logits, self.k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        out = torch.zeros_like(x_flat)

        # 简单的循环实现 (生产环境建议用 CUDA kernel 优化)
        for i in range(self.k):
            idx = indices[:, i]  # (batch*seq,)
            w = weights[:, i].unsqueeze(1)  # (batch*seq, 1)

            for e_idx in range(self.num_experts):
                mask = (idx == e_idx)
                if mask.sum() > 0:
                    # 只计算被分配到当前专家的 token
                    expert_in = x_flat[mask]
                    expert_out = self.experts[e_idx](expert_in)
                    out[mask] += w[mask] * expert_out

        return out.view(batch, seq, dim)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.c_attn = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.c_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
                             .view(1, 1, cfg.max_seq_len, cfg.max_seq_len))

    def forward(self, x):
        B, T, C = x.size()  # C 这里就是 cfg.dim

        # 修改点在这里：把 cfg.dim 改成了 C
        q, k, v = self.c_attn(x).split(C, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.dim)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.dim)
        self.moe = MoELayer(cfg)  # 这里用 MoE 替换了普通 MLP

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class BabyGrok(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss