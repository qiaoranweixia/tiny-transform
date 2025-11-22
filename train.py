import os
import torch
import numpy as np
from config import ModelConfig
from model import BabyGrok

# 1. 设置
cfg = ModelConfig()
os.makedirs('out', exist_ok=True)
print(f"Training on {cfg.device}...")

# 2. 加载数据
data_path = os.path.join('data', 'train.bin')
if not os.path.exists(data_path):
    raise FileNotFoundError("请先运行 prepare_data.py")

# 使用 memmap 内存映射读取大文件，节省内存
data = np.memmap(data_path, dtype=np.uint16, mode='r')


def get_batch():
    ix = torch.randint(len(data) - cfg.max_seq_len, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + cfg.max_seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + cfg.max_seq_len]).astype(np.int64)) for i in ix])
    return x.to(cfg.device), y.to(cfg.device)


# 3. 初始化模型
model = BabyGrok(cfg)
model.to(cfg.device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# 4. 训练循环
model.train()
for step in range(cfg.max_iters):
    # 获取数据
    xb, yb = get_batch()

    # 前向计算
    logits, loss = model(xb, yb)

    # 反向传播
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # 打印日志
    if step % cfg.eval_interval == 0:
        print(f"Step {step}: Loss {loss.item():.4f}")

# 5. 保存模型
torch.save(model.state_dict(), os.path.join('out', 'model.pth'))
print("训练结束，模型已保存到 out/model.pth")