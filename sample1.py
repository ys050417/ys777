import os
import torch
from model import GPTConfig, GPT

# 配置参数
out_dir = 'out-poemtext-char'  # 你的模型保存目录
device = 'cuda' if torch.cuda.is_available() else 'cpu'
start = "\n"  # 起始文本（空行）
num_samples = 10  # 生成诗词数量
max_new_tokens = 500  # 单首诗词最大长度
temperature = 0.8  # 生成随机性（0更保守，1更随机）
top_k = 200  # 采样范围

# 加载模型
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# 加载词汇表（适配你的数据集）
meta_path = os.path.join('data', 'poemtext', 'meta.pkl')
if os.path.exists(meta_path):
    import pickle
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    #  fallback to gpt2 tokenizer
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s)
    decode = lambda l: enc.decode(l)

# 编码起始文本
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# 生成诗词
print("生成唐诗：")
print("="*50)
with torch.no_grad():
    for k in range(num_samples):
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(decode(y[0].tolist()))
        print('-'*50)