import os
from contextlib import nullcontext

import numpy as np
import time
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

from config import TrainConfig
from model import Config, GPT

cfg = TrainConfig()
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
cfg.out_dir.mkdir(parents=True, exist_ok=True)
enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab

def encode(s):
    return enc.encode(s)
def decode(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return enc.decode(ids)

train_data = np.memmap(cfg.data_dir / "train.bin", dtype=np.uint16, mode="r")
val_data   = np.memmap(cfg.data_dir / "val.bin",   dtype=np.uint16, mode="r")

def get_batch(split):
    data = train_data if split== 'train' else val_data
    ix = torch.randint(len(data)-cfg.block_size-1,(cfg.batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i+cfg.block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i+1:i+cfg.block_size+1].astype(np.int64)) for i in ix
    ])
    return x.to(device), y.to(device)

def get_lr(it):
    if it < warmup_iters:
        return cfg.lr * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)

def setup_ddp():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if not ddp:
        return False, 0, 0, 1, "cuda:0" if torch.cuda.is_available() else "cpu"

    dist.init_process_group(backend="nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size, device

ddp, rank, local_rank, world_size, device = setup_ddp()
torch.manual_seed(cfg.seed + rank)
torch.cuda.manual_seed_all(cfg.seed + rank)
master = (rank == 0)
tokens_per_iter = cfg.batch_size * cfg.block_size * world_size
max_iters = math.ceil(cfg.target_tokens / tokens_per_iter)
warmup_iters = max(1000, int(0.01 * max_iters))
lr_decay_iters = max_iters
eval_interval = max(1000, max_iters // 200)
eval_iters = cfg.eval_iters
save_interval = cfg.save_interval

device_type = "cuda" if "cuda" in device else "cpu"
dtype = "bfloat16" if (device_type=="cuda" and torch.cuda.is_bf16_supported()) else "float32"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type=="cuda" else nullcontext()

config = Config(
    vocab_size=vocab_size,
    block_size=cfg.block_size,
    n_layer=cfg.n_layer,
    n_head=cfg.n_head,
    n_embd=cfg.n_embd,
    dropout=cfg.dropout,
    bias=cfg.bias
)

model = GPT(config)
model.to(device)


if ddp:
    ddp_model = DDP(model, device_ids=[local_rank])
else:
    ddp_model = model

model =  ddp_model.module if ddp else model

optimizer = model.config_optimizer(
    decay_weight=cfg.weight_decay,
    lr=cfg.lr,
    betas=cfg.betas,
    device_type=device_type
)

start_iter = 0
if cfg.resume_path.exists():
    obj = torch.load(cfg.resume_path, map_location=device)

    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"])
        optimizer.load_state_dict(obj["optimizer"])
        start_iter = obj["iter"] + 1
        if master or not ddp:
            print(f"Resumed FULL ckpt from {cfg.resume_path} at iter={obj['iter']}", flush=True)

    else:
        model.load_state_dict(obj)
        start_iter = 28000 + 1
        if master or not ddp:
            print(f"Resumed MODEL-only ckpt from {cfg.resume_path} (no optimizer). start_iter={start_iter}", flush=True)

@torch.no_grad()
def evaluate():
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X,Y = get_batch('val')
        with ctx:
            logits, loss = model(X, Y)
        losses[k] = loss.item()

    model.train()
    return losses.mean()

@torch.no_grad()
def sample():
    model.eval()
    start = "Hello, I am"
    idx = torch.tensor([encode(start)], device=device)
    idx = model.generate(
        idx,
        max_new_tokens=200,
        temp=1.0,
        top_k=40
    )
    print("\n=== SAMPLE ===")
    print(decode(idx[0]))
    print("==============\n")
    model.train()
t0 = time.time()
for iter in range(start_iter,max_iters):
    learn_rate = get_lr(iter) if cfg.decay_lr else cfg.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = learn_rate

    X,Y = get_batch("train")
    with ctx:
        logits, loss = ddp_model(X,Y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    optimizer.step()
    if iter % 10 == 0:
        if master or not ddp:
            print(f"iter {iter}, loss {loss.item():.4f}")
    if iter % eval_interval == 0 and iter>0:
        if master or not ddp:
            val_loss = evaluate()
            print(f"\nVAL loss: {val_loss:.4f}")
            sample()
            if iter % save_interval==0:
                torch.save({"model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iter": iter}, cfg.checkpoint_dir / f"model_it={iter}.pt")
                print("model saved\n")


if ddp:
    dist.destroy_process_group()

if master or not ddp:
    torch.save(model.state_dict(), cfg.out_dir / "final_model.pt")

t1 = time.time()
print("training finished in", t1-t0, "seconds")
