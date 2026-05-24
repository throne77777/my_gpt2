import os
from contextlib import nullcontext

import numpy as np
import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from config import SFTConfig
from model import Config, GPT

cfg = SFTConfig()
cfg.out_dir.mkdir(parents=True, exist_ok=True)
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
# os.makedirs(out_dir, exist_ok=True)

enc = tiktoken.get_encoding('gpt2')
vocab_size = enc.n_vocab
torch.manual_seed(cfg.seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"

dtype = "bfloat16" if (device_type=="cuda" and torch.cuda.is_bf16_supported()) else "float32"
ptdtype = {"float32": torch.float32,
           "bfloat16": torch.bfloat16,
           "float16": torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type=="cuda" else nullcontext()

def encode(s):
    return enc.encode(s)
def decode(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return enc.decode(ids)

train_data = torch.load(cfg.data_dir / "train_sft_dolly.pt")
val_data   = torch.load(cfg.data_dir / "val_sft_dolly.pt")
def get_batch(split):
    data = train_data if split== 'train' else val_data
    ix = torch.randint(len(data),(cfg.batch_size,))
    x = torch.stack([
        data[i]["input_ids"] for i in ix
    ])
    y = torch.stack([
        data[i]["labels"] for i in ix
    ])
    return x.to(device), y.to(device)
#右移操作依然存在，不过隐式发生（在训练过程中发生）

def get_lr(it):
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)

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
model.load_state_dict(torch.load(cfg.resume_path, map_location="cpu"),strict=True)
model.to(device)

optimizer = model.config_optimizer(
    decay_weight=cfg.weight_decay,
    lr=cfg.lr,
    betas=cfg.betas,
    device_type=device_type
)

@torch.no_grad()
def evaluate():
    model.eval()
    losses = torch.zeros(cfg.eval_iters)
    for k in range(cfg.eval_iters):
        X,Y = get_batch('val')
        with ctx:
            logits, loss = model(X, Y, True)
        losses[k] = loss.item()

    model.train()
    return losses.mean()

@torch.no_grad()
def sample():
    model.eval()
    start = "system:\nYou are a helpful assistant.\n\nuser:\nHello.\n\nassistant:\n"
    idx = torch.tensor([encode(start)], device=device)
    idx = model.generate(
        idx,
        max_new_tokens=200,
        temp=1.0,
        top_k=40,
        eos_token_id= enc.eot_token
    )
    print("\n=== SAMPLE ===")
    print(decode(idx[0]))
    print("==============\n")
    model.train()
t0 = time.time()
for iter in range(cfg.max_iters):
    learn_rate = get_lr(iter) if cfg.decay_lr else cfg.lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = learn_rate

    X,Y = get_batch("train")
    with ctx:
        logits, loss = model(X,Y,True)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    #prevent the gradient becoming too large
    optimizer.step()
    if iter % 10 == 0:
        print(f"iter {iter}, loss {loss.item():.4f}")
    if iter % cfg.eval_interval == 0 and iter>0:
        val_loss = evaluate()
        print(f"\nVAL loss: {val_loss:.4f}")
        sample()
        if iter % cfg.save_interval == 0:
            torch.save({"model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "iter": iter}, cfg.out_dir / f"sft_it={iter}.pt")
            print("model saved\n")

torch.save(model.state_dict(), cfg.checkpoint_dir / "final_model.pt")

t1 = time.time()
print("training finished in", t1-t0, "seconds")
