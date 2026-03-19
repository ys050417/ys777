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