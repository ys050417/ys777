# nanoGPT

## 一、研究简介

​	当前主流生成式预训练 Transformer（GPT）模型普遍存在参数量庞大、工程实现复杂的问题，极大提升初学者的学习与复现门槛。nanoGPT 项目是基于 PyTorch 框架实现的轻量化 GPT 复现方案，完整覆盖模型训练与推理流程，其核心设计目标为构建小巧、简洁、可解释的教育向大语言模型原型。若将工业级 GPT 类比为 “航空母舰”，nanoGPT 则可视为功能完备的 “微型游艇”，在保留 Transformer 核心机制的同时显著降低实践成本，对大模型技术的入门教学具有重要价值。

本研究基于三类数据集开展模型训练与验证：

- 一是复现nanoGPT莎士比亚实验；

- 二是包含 58000 首诗词的中文诗歌语料库，用于训练歌词生成模型；

- 三是约 124 万字的《天龙八部》小说文本，用于训练《天⻰⼋部》⻛格化的生成模型。

## 二、复现nanoGPT莎士比亚实验

### （一）安装相关库

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

### （二）生成数据集

- 生成命令

```
python data/shakespeare_char/prepare.py
```

- 生成结果

![image-20260323092745955](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260323092745955.png)

### （三）训练集和测试集划分

- 按照训练集和测试集9:1进行划分
- prepare.py

```python
import os
import pickle
import requests
import numpy as np

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
```

- 划分结果

  ![image-20260323093932471](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260323093932471.png)

### （四）训练数据集

- 训练命令

```
python train.py config/train_shakespeare_char.py
```

- 参数说明

|         参数名         |        默认值        |                         参数说明                          |                           调整效果                           |
| :--------------------: | :------------------: | :-------------------------------------------------------: | :----------------------------------------------------------: |
|        out_dir         | out-shakespeare-char |   模型训练过程中生成的 checkpoint、日志等文件的输出目录   |                         修改存储路径                         |
|     eval_interval      |         250          |         每隔多少步（iter）评估一次训练 / 验证损失         | 调大：减少评估次数，训练速度更快，但过拟合风险难以及时发现。调小：更频繁监控损失变化，能更早发现过拟合 / 欠拟合，但会增加训练耗时 |
|      log_interval      |          10          |     每隔多少步打印一次训练日志（loss、时间、MFU 等）      | 调大：减少日志打印频率，避免刷屏，适合长时间训练调小：更密集查看训练状态，适合调试阶段 |
|       eval_iters       |         200          | 每次评估时，在训练 / 验证集上各采样多少个批次计算平均损失 | 调大：评估结果更稳定、准确，但评估耗时更长。调小评估更快，但损失值波动更大，结果不够可靠 |
|       eval_only        |        False         |             是否仅加载模型做评估，不进行训练              |      True：仅验证已有模型性能，适合快速测试 checkpoint       |
| always_save_checkpoint |        False         |               是否每次评估都保存 checkpoint               | `True`：保存所有 checkpoint，便于回溯训练过程，但占用更多磁盘空间； `False`：仅保存验证损失最优的 checkpoint，节省磁盘空间 |
|       init_from        |       scratch        |                      模型初始化方式                       | `scratch`：从零开始训练； `resume`：从`out_dir`中的 checkpoint 恢复训练； `gpt2`/`gpt2-small`等：加载 OpenAI 预训练 GPT-2 权重初始化 |

- train.py文件

```python
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 新手适配：莎士比亚字符级训练配置（轻量化）
# I/O
out_dir = 'out-shakespeare-char'  # 模型输出目录
eval_interval = 250  # 频繁评估（小数据集易过拟合）
log_interval = 10    # 减少打印频率，避免刷屏
eval_iters = 200     # 评估迭代数
eval_only = False    # 仅评估？
always_save_checkpoint = False  # 仅保存最优模型，减少磁盘占用
init_from = 'scratch' # 从零训练
# wandb logging
wandb_log = False 
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'
# data
dataset = 'shakespeare_char'  
gradient_accumulation_steps = 1  # 关闭梯度累积
batch_size = 16  # 批次大小（根据显存调整，小显存可改8）
block_size = 256 # 上下文长度（轻量化）
# model
n_layer = 6      # 网络层数（迷你版GPT）
n_head = 6       # 注意力头数
n_embd = 384     # 嵌入维度
dropout = 0.2    # dropout率（防止过拟合）
bias = False     # 偏置
# adamw optimizer
learning_rate = 1e-3  # 学习率（小模型可稍高）
max_iters = 5000      # 训练总步数（快速验证）
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99     # 小数据集调大beta2
grad_clip = 1.0  # 梯度裁剪
# learning rate decay settings
decay_lr = True
warmup_iters = 100 # 预热步数（小数据集减少）
lr_decay_iters = 5000 # 衰减步数=总步数
min_lr = 1e-4    # 最小学习率
# DDP settings
backend = 'nccl'
# system
device = 'cuda' if torch.cuda.is_available() else 'cpu'  
dtype = 'float16' if torch.cuda.is_available() else 'float32'  
compile = False   
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
# 关键修改：适配新版PyTorch，消除GradScaler警告
if device_type == 'cuda' and dtype == 'float16':
    scaler = torch.amp.GradScaler('cuda', enabled=True)
else:
    scaler = torch.amp.GradScaler('cuda', enabled=False)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
```

- 部分运行截图

![image-20260323094158801](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260323094158801.png)

### （五）利用模型生成文本

- sample.py

```python
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out-shakespeare-char' # 关键修改：匹配训练的模型目录
start = "\nROMEO:" # 自定义开头：罗密欧的台词
num_samples = 3 # 生成3段文本
max_new_tokens = 300 # 每段生成300个字符
temperature = 0.7 # 降低随机性，更贴合莎士比亚风格
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 自动检测设备
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # 关闭编译，避免兼容问题
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    # 关键修改：添加weights_only=True消除警告
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
```

- 生成结果

```
ROMEO:
No; but this is a word of blood: I will say
I am grave to make us barriage out of her back:
It was, and that I forget my bones of the officer
Where is dispatch'd to the hour of it.

BENVOLIO:
Be it possible was to lose him; he lies
He comes at his enemy.

ROMEO:
Here comes this deserves the like ag
---------------

ROMEO:
To Warwick, there's a traitor of Edward's face,
Let me forget the morning of sorrow.

WARWICK:
I know not the duke of Lancaster;
But where he shall please him to him that have tell him
And from his conquest of his hateful son
When he hath made me not between to take him
When he will success the fri
---------------

ROMEO:
I fear the people, madam. Why do you suppose
In a better than may say to the prince?

ROMEO:
The hour of Romeo, not at Claudio,
And then in the news of the wars bear
Of the commonwealth of the self-moon of collars?

JULIET:
Heavens, how I shall post thee, do not your grace?

Nurse:
To heaven, sir, 
---------------
```

## 三、歌词生成模型实验

### （一）下载唐诗数据集

![image-20260323102201251](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260323102201251.png)

### （二）划分数据集和测试集

- 按照9:1的比例将唐诗数据集划分为训练集和测试集
- prepare.py

```python
import os
import chardet
import tiktoken
import numpy as np

input_file_path = r'tangshi.txt'

#检测文件编码
with open(input_file_path, 'rb') as f:
    raw_data = f.read()
    result = chardet.detect(raw_data)
    file_encoding = result['encoding']
    print(f"检测到文件编码：{file_encoding}")

#用检测到的编码读取文件
with open(input_file_path, 'r', encoding=file_encoding, errors='ignore') as f:
    data = f.read()

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
```

- 划分结果

![image-20260323102220629](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260323102220629.png)

### （三）训练数据集

- train1.py

```python
import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# 关键修复1：抑制dynamo编译错误
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# 确保能导入model模块
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# 核心配置：关闭编译 + 适配唐诗数据集
# I/O
out_dir = 'out-poemtext-char'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'
# wandb日志
wandb_log = False
wandb_project = 'poemtext-char'
wandb_run_name = 'nanoGPT-poem'
# 数据配置
dataset = 'poemtext'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
# 模型配置
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False
# 优化器配置
learning_rate = 1e-3
max_iters = 5000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0
# 学习率衰减
decay_lr = True
warmup_iters = 100
lr_decay_iters = 5000
min_lr = 1e-4
# DDP配置
backend = 'nccl'
# 系统配置：关键修复2 - 强制关闭编译
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'float16' if torch.cuda.is_available() else 'float32'
compile = False  # 强制关闭编译，无论是否GPU
# -----------------------------------------------------------------------------
# CPU参数适配
if device == 'cpu':
    batch_size = 12
    block_size = 64
    n_layer = 4
    n_head = 4
    n_embd = 128
    dropout = 0.0
    max_iters = 2000
    lr_decay_iters = 2000
    eval_iters = 20
# -----------------------------------------------------------------------------
# 加载配置覆盖
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
if os.path.exists('configurator.py'):
    exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

# DDP初始化
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

# 打印配置
if master_process:
    print(f"===== 训练配置 =====")
    print(f"设备: {device}")
    print(f"数据集: {dataset}")
    print(f"批次大小: {batch_size}, 上下文长度: {block_size}")
    print(f"模型层数: {n_layer}, 嵌入维度: {n_embd}")
    print(f"训练总步数: {max_iters}")
    print(f"编译模式: {compile} (已强制关闭)")
    print(f"====================")

# 输出目录创建
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    os.makedirs(out_dir, exist_ok=True)
    print(f"每次迭代token数: {tokens_per_iter:,}")

# 随机种子
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'

# 混合精度上下文
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# 数据加载函数
data_dir = os.path.join('data', dataset)
def get_batch(split):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# 模型初始化
iter_num = 0
best_val_loss = 1e9

# 加载词汇表大小
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"加载词汇表大小: {meta_vocab_size}")

# 模型参数
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)

# 从零初始化模型
if init_from == 'scratch':
    print("从零初始化唐诗生成模型...")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"从 {out_dir} 恢复训练...")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"从GPT-2预训练权重初始化: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

# 裁剪上下文长度
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# 梯度缩放器
if device_type == 'cuda' and dtype == 'float16':
    scaler = torch.amp.GradScaler('cuda', enabled=True)
else:
    scaler = torch.amp.GradScaler('cuda', enabled=False)

# 优化器
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # 释放内存

# 关键修复3：移除模型编译逻辑（不再编译）
print("跳过模型编译（避免Triton依赖）...")

# DDP包装
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# 损失评估函数
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 学习率调度函数
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# wandb日志初始化
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# 训练主循环
print("开始训练唐诗生成模型...")
X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    # 设置当前学习率
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # 评估并保存模型
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"步骤 {iter_num}: 训练损失 {losses['train']:.4f}, 验证损失 {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"保存模型到 {out_dir}/ckpt.pt")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # 前向/反向传播
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        # 预取下一批数据
        X, Y = get_batch('train')
        # 反向传播
        scaler.scale(loss).backward()

    # 梯度裁剪
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # 优化器更新
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # 日志打印
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"迭代 {iter_num}: 损失 {lossf:.4f}, 耗时 {dt*1000:.2f}ms, MFU {running_mfu*100:.2f}%" if running_mfu != -1.0 else f"迭代 {iter_num}: 损失 {lossf:.4f}, 耗时 {dt*1000:.2f}ms")

    # 迭代计数
    iter_num += 1
    local_iter_num += 1

    # 终止条件
    if iter_num > max_iters:
        break

# 清理DDP
if ddp:
    destroy_process_group()

print("训练完成！模型已保存到", out_dir)
```

- 部分训练截图

![image-20260323103038773](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260323103038773.png)

### （四）利用训练的模型生成歌词 

- sample1.py

```python
import os
import torch
from model import GPTConfig, GPT

# 配置参数
out_dir = 'out-poemtext-char'  # 模型保存目录
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

# 加载词汇表
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
```

- 生成唐诗结果

![image-20260323103525984](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260323103525984.png)

## 四、《天⻰⼋部》⻛格化的模型实验

### （一）下载《天龙八部》数据集

![image-20260323103906835](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260323103906835.png)

### （二）划分数据集和测试集

- 按照9:1的比例将《天龙八部》数据集划分为训练集和测试集
- prepare.py

```python
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
```

- 划分结果

![image-20260323104151233](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260323104151233.png)

### （三）训练数据集

- train3.py

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Torch was not compiled with flash attention.*")
import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from model import GPTConfig, GPT

config = {
    "out_dir": "out-tianlong-char",
    "eval_interval": 250,
    "eval_iters": 200,
    "log_interval": 1,
    "dataset": "tianlong",
    "gradient_accumulation_steps": 1,
    "batch_size": 16,
    "block_size": 128,
    "n_layer": 6,
    "n_head": 8,
    "n_embd": 256,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "max_iters": 5000,
    "lr_decay_iters": 5000,
    "min_lr": 1e-4,
    "beta2": 0.99,
    "warmup_iters": 100,
    # GPU 兼容配置
    "init_from": "scratch",
    "eval_only": False,
    "always_save_checkpoint": True,
    "bias": False,
    "weight_decay": 1e-1,
    "beta1": 0.9,
    "grad_clip": 1.0,
    "decay_lr": True,
    "compile": False,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dtype": "float32",}


# -----------------------------------------------------------------------------
# 工具函数
def get_lr(it):
    """学习率衰减策略"""
    if it < config["warmup_iters"]:
        return config["learning_rate"] * it / config["warmup_iters"]
    if it > config["lr_decay_iters"]:
        return config["min_lr"]
    decay_ratio = (it - config["warmup_iters"]) / (config["lr_decay_iters"] - config["warmup_iters"])
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])


def load_data():
    """加载预处理后的二进制数据"""
    data_dir = os.path.join("data", config["dataset"])
    # 检查数据文件是否存在
    train_bin_path = os.path.join(data_dir, "train.bin")
    val_bin_path = os.path.join(data_dir, "val.bin")
    if not os.path.exists(train_bin_path) or not os.path.exists(val_bin_path):
        raise FileNotFoundError(
            f"未找到预处理数据！请先运行 data/tianlong/prepare.py 生成 {train_bin_path} 和 {val_bin_path}")

    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(val_bin_path, dtype=np.uint16, mode="r")

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - config["block_size"], (config["batch_size"],))
        x = torch.stack([torch.from_numpy((data[i:i + config["block_size"]]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + config["block_size"]]).astype(np.int64)) for i in ix])
        x, y = x.to(config["device"]), y.to(config["device"])
        return x, y

    return get_batch


# -----------------------------------------------------------------------------
# 主训练逻辑
if __name__ == "__main__":
    # GPU 环境初始化
    torch.manual_seed(1337)
    torch.cuda.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in config["device"] else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config["dtype"]]
    ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # 加载数据
    get_batch = load_data()

    # 初始化模型（参数已满足整除要求）
    model_args = dict(
        n_layer=config["n_layer"],
        n_head=config["n_head"],
        n_embd=config["n_embd"],
        block_size=config["block_size"],
        bias=config["bias"],
        vocab_size=50304,  # GPT-2 词汇表大小
        dropout=config["dropout"]
    )
    print(f"初始化 GPU 模型 | 配置：n_layer={config['n_layer']}, n_head={config['n_head']}, n_embd={config['n_embd']}")
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf).to(config["device"])

    # 优化器配置
    optimizer = model.configure_optimizers(
        weight_decay=config["weight_decay"],
        learning_rate=config["learning_rate"],
        betas=(config["beta1"], config["beta2"]),
        device_type=device_type
    )

    # 训练循环
    model.train()
    X, Y = get_batch("train")
    t0 = time.time()
    local_iter_num = 0
    raw_model = model
    running_mfu = -1.0

    print(
        f"\n开始 GPU 训练 | 数据集：{config['dataset']} | 总迭代数：{config['max_iters']} | 批次大小：{config['batch_size']}")
    print("=" * 80)

    for iter_num in range(config["max_iters"]):
        # 动态调整学习率
        lr = get_lr(iter_num) if config["decay_lr"] else config["learning_rate"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # 定期验证
        if iter_num % config["eval_interval"] == 0:
            model.eval()
            losses = torch.zeros(config["eval_iters"])
            for k in range(config["eval_iters"]):
                X_val, Y_val = get_batch("val")
                with ctx:
                    logits, loss = model(X_val, Y_val)
                losses[k] = loss.item()
            loss_val = losses.mean()
            print(f"\n【验证】迭代 {iter_num} | 验证损失：{loss_val:.4f} | 当前学习率：{lr:.4e}")
            model.train()

        # 保存模型 checkpoint
        if iter_num % config["eval_interval"] == 0 and config["always_save_checkpoint"]:
            os.makedirs(config["out_dir"], exist_ok=True)
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "config": config,
            }
            ckpt_path = os.path.join(config["out_dir"], "ckpt.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"【保存】模型已保存至 {ckpt_path}")

        # 训练步骤
        for micro_step in range(config["gradient_accumulation_steps"]):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / config["gradient_accumulation_steps"]
            loss.backward()

        # 梯度裁剪
        if config["grad_clip"] != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])

        # 更新参数
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # 打印训练日志
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % config["log_interval"] == 0:
            lossf = loss.item() * config["gradient_accumulation_steps"]
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(config["batch_size"] * config["gradient_accumulation_steps"], dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f"【训练】迭代 {iter_num} | 训练损失：{lossf:.4f} | 耗时：{dt * 1000:.2f}ms | GPU 利用率：{running_mfu * 100:.2f}%" if running_mfu != -1.0 else f"【训练】迭代 {iter_num} | 训练损失：{lossf:.4f} | 耗时：{dt * 1000:.2f}ms")

        # 加载下一个批次
        X, Y = get_batch("train")
        local_iter_num += 1

    # 保存最终模型
    final_ckpt_path = os.path.join(config["out_dir"], "final_ckpt.pt")
    torch.save({
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": config["max_iters"],
        "config": config,
    }, final_ckpt_path)
    print("\n" + "=" * 80)
    print(f"GPU 训练完成！最终模型已保存至 {final_ckpt_path}")
```

- 部分训练截图

![image-20260323104747488](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260323104747488.png)

### （四）利用训练的模型生成《天⻰⼋部》⻛格化文本

- sample.py

```python
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
```

- 生成结果

![image-20260323105247533](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20260323105247533.png)