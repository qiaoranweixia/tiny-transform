import torch
import tiktoken
from config import ModelConfig
from model import BabyGrok
import os

# 1. 加载配置和模型
cfg = ModelConfig()
model = BabyGrok(cfg)
ckpt_path = os.path.join('out', 'model.pth')

if os.path.exists(ckpt_path):
    print(f"加载模型权重: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=cfg.device))
else:
    print("未找到模型权重，将使用随机权重演示...")

model.to(cfg.device)
model.eval()

# 2. 准备输入
enc = tiktoken.get_encoding("gpt2")
start_text = "Once upon a time"
input_ids = enc.encode(start_text)
x = torch.tensor(input_ids, dtype=torch.long, device=cfg.device).unsqueeze(0)

# 3. 生成
print(f"\n输入: {start_text}\n")
print("Grok 回答: ", end="", flush=True)

with torch.no_grad():
    for _ in range(100):  # 生成 100 个 token
        # 截断输入
        x_cond = x[:, -cfg.max_seq_len:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :]

        # 采样
        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_idx = torch.multinomial(probs, num_samples=1)

        # 拼接
        x = torch.cat((x, next_idx), dim=1)

        # 打印字符
        print(enc.decode([next_idx.item()]), end="", flush=True)

print("\n")