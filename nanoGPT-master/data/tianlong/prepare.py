import os
import tiktoken
import numpy as np

# 1. 读取文本
input_file_path = "tianlong.txt"
with open(input_file_path, "r", encoding="utf-8") as f:
    data = f.read()

# 2. 9:1 切分训练/验证集
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# 3. GPT-2 BPE 编码
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train tokens: {len(train_ids)}, val tokens: {len(val_ids)}")

# 4. 保存为二进制文件
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.dirname(__file__) + "/train.bin")
val_ids.tofile(os.path.dirname(__file__) + "/val.bin")