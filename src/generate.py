import os
import torch
import tiktoken
from model import Config, GPT
from pathlib import Path

# ====== paths ======
root = Path(__file__).resolve().parent
out_dir = root.parent / "train_out"
ckpt_path = root.parent / "train_ckpt"

# ====== same config as training ======
enc = tiktoken.get_encoding("gpt2")
block_size = 1024
vocab_size = enc.n_vocab
n_layer = 12
n_embd = 768
n_head = 12
dropout = 0.1
bias = True

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)

def encode(s: str):
    return enc.encode(s)

def decode(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return enc.decode(ids)

def main():
    # 1) build model
    config = Config(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )
    model = GPT(config).to(device)

    # 2) load checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()

    # 3) prompt: Who are you?
    start = "I'm"
    idx = torch.tensor([encode(start)], device=device)

    # 4) generate
    with torch.no_grad():
        idx_out = model.generate(
            idx,
            max_new_tokens=200,
            temp=0.8,
            top_k=40,
            eos_token_id=enc.eot_token,  # gpt2 的 eot id = 50256
        )

    print("=== PROMPT ===")
    print(start)
    print("=== OUTPUT ===")
    print(decode(idx_out[0]))
    print("==============")

if __name__ == "__main__":
    main()