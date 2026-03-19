import os
import torch
import tiktoken
from model import GPTConfig, GPT


out_dir = "out-tianlong-char"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = "float32"
start_text = "段誉踏入无量山，只见"  # 开头文本
max_new_tokens = 600  # 文本长度
temperature = 0.7  # 生成随机性
top_k = 40  # 采样范围

# 加载模型配置和权重
ckpt_path = os.path.join(out_dir, "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)
config = checkpoint["config"]
model_args = dict(
    n_layer=config["n_layer"],
    n_head=config["n_head"],
    n_embd=config["n_embd"],
    block_size=config["block_size"],
    bias=config["bias"],
    vocab_size=50304,
    dropout=config["dropout"]
)
gptconf = GPTConfig(**model_args)
model = GPT(gptconf)
model.load_state_dict(checkpoint["model"])
model.to(device)
model.eval()

# 编码工具
enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s)
decode = lambda l: enc.decode(l)

# 编码起始文本
start_ids = encode(start_text)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# 生成文本
with torch.no_grad():
    y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    generated_text = decode(y[0].tolist())
    print("\n" + "="*50 + " 生成结果 " + "="*50)
    print(generated_text)
    print("="*110 + "\n")