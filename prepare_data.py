import os
import tiktoken
import numpy as np
from datasets import load_dataset

# 1. 下载数据 (使用 HuggingFace 的 TinyStories)
print("正在下载 TinyStories 数据集...")
dataset = load_dataset("roneneldan/TinyStories", split="train[:20000]") # 为了演示只取前2万条

# 2. 准备 Tokenizer
print("正在编码数据...")
enc = tiktoken.get_encoding("gpt2")

# 3. 编码所有文本
data = []
for item in dataset:
    ids = enc.encode_ordinary(item['text'])
    data.extend(ids)
    data.append(enc.eot_token) # 每个故事结束加一个结束符

# 4. 转换为 numpy 数组并保存
print(f"总 Token 数: {len(data)}")
data_np = np.array(data, dtype=np.uint16)

os.makedirs('data', exist_ok=True)
data_np.tofile(os.path.join('data', 'train.bin'))
print("数据处理完成！已保存到 data/train.bin")
