from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    root: Path = Path(__file__).resolve().parent
    data_dir: Path = root.parent / "data" / "openwebtext"
    out_dir: Path = root.parent / "train_out"
    checkpoint_dir: Path = root.parent / "train_ckpt"
    resume_path: Path = out_dir / "model_it=28000.pt"
    target_tokens: int = int(120_000_000 * 26)
    block_size: int = 1024
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    dropout: float = 0.1
    bias: bool = True
    batch_size: int = 32
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    sample_tokens: int = 200
    temperature: float = 0.9
    top_k: int = 50
    grad_clip: float = 1.0
    decay_lr: bool = True
    eval_iters: int = 50
    save_interval: int = 5000
    seed: int = 1337


@dataclass
class SFTConfig:
    root: Path = Path(__file__).resolve().parent
    data_dir: Path = root.parent / "data" / "sft_af_opt"
    out_dir: Path = root.parent / "sft_out"
    checkpoint_dir: Path = root.parent / "sft_ckpt"
    resume_path: Path = root.parent / "train_out" / "final_model.pt"
    target_tokens: int = 0
    block_size: int = 1024
    n_layer: int = 12
    n_embd: int = 768
    n_head: int = 12
    dropout: float = 0.1
    bias: bool = True
    batch_size: int = 32
    lr: float = 1e-5
    min_lr: float = 1e-6
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    sample_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40
    grad_clip: float = 1.0
    decay_lr: bool = True
    eval_iters: int = 50
    save_interval: int = 250
    seed: int = 1337
    warmup_iters: int = 50
    max_iters: int = 600
    lr_decay_iters: int = 600
    eval_interval: int = 100
