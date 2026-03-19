import os
import chardet
import tiktoken
import numpy as np

# 先安装依赖（如果没装的话）
# pip install chardet

input_file_path = r'tangshi.txt'

# 第一步：检测文件编码
with open(input_file_path, 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    file_encoding = result['encoding']
    print(f"检测到文件编码：{file_encoding}")

# 第二步：用检测到的编码读取文件
with open(input_file_path, 'r', encoding=file_encoding, errors='ignore') as f:
    data = f.read()

# 后续代码不变
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)

print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))