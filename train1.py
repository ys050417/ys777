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