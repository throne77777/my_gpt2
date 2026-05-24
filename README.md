# my_gpt2

A from-scratch implementation of a GPT-2 style language model (**~123.65M parameters**), built with full pretraining and supervised fine-tuning (SFT) pipelines.

This project is implemented **largely from scratch with PyTorch**, inspired by the design philosophy of **NanoGPT**, while reproducing and extending the complete GPT-2 training workflow:

- GPT-2 style decoder-only Transformer implementation
- OpenWebText pretraining
- Instruction supervised fine-tuning (SFT)
- Distributed training (DDP / cluster-ready)
- Text generation and checkpoint recovery

Unlike highly abstracted training frameworks, this repository is designed to be **readable, modifiable, and research-friendly**, making the full language model training pipeline transparent.

---

## Highlights

- Built a **GPT-2 Small scale model (~123.65M parameters)** largely from scratch
- Implemented a **12-layer, 12-head, 768 hidden dimension Transformer**
- Completed the **full pipeline from pretraining to SFT**
- Trained on a **subset of OpenWebText** using autoregressive next-token prediction
- Fine-tuned on **instruction-following datasets (Dolly / Alpaca style)**
- Supports **checkpointing, resume training, sampling, and DDP multi-GPU training**
- Research-oriented codebase for **understanding and experimenting with LLM training**

---

## Model Scale

Current default configuration approximately matches **GPT-2 Small**:

- **Layers:** `12`
- **Attention Heads:** `12`
- **Hidden Size:** `768`
- **Context Length:** `1024`
- **Parameters:** `~123.65M`

---

## Project Goal

The goal of this project is to **deeply understand and reproduce the training pipeline of GPT-style language models**, rather than relying on high-level frameworks.

The repository covers the complete lifecycle of a language model:

```text
model implementation → data preparation → pretraining → SFT → generation
```

This project emphasizes:

- Understanding Transformer internals
- Building training pipelines from scratch
- Reproducible experimentation
- Easy architectural modification

---

## Features

### 1. From-scratch GPT-2 Implementation

Implemented core GPT-2 components manually in PyTorch:

- Multi-head causal self-attention
- Transformer blocks
- Pre-LayerNorm architecture
- Residual connections
- Weight tying
- AdamW optimizer parameter grouping
- Causal masking

---

### 2. OpenWebText Pretraining

Pretraining pipeline includes:

- OpenWebText tokenization
- Binary `memmap` dataset storage
- Random context window sampling
- Warmup + cosine learning rate scheduling
- Checkpoint saving and resume training

---

### 3. Supervised Fine-Tuning (SFT)

Implemented instruction tuning with:

- Dolly / Alpaca style datasets
- Chat-style prompt formatting
- Label masking (**assistant-only loss**)
- Pretrained checkpoint initialization

---

### 4. Distributed Training

Supports multi-GPU training through **PyTorch DDP**.

Example launch (Linux + CUDA multi-GPU environment):

```bash
torchrun --nproc_per_node=<num_gpus> src/train.py
```

Also suitable for cluster environments (e.g., Slurm).

---

## Repository Structure

```text
my_gpt2/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ config.py
│  ├─ model.py
│  ├─ train.py
│  ├─ sft.py
│  └─ generate.py
├─ data/
│  ├─ openwebtext/
│  ├─ shakespeare/
│  ├─ sft/
│  └─ sft_af_opt/
├─ train_ckpt/
├─ train_out/
├─ sft_ckpt/
└─ sft_out/
```

---

## Tech Stack

- Python 3.10
- PyTorch
- CUDA
- DDP (`torchrun`)
- `tiktoken`
- HuggingFace Datasets

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Prepare Pretraining Data

Prepare OpenWebText for autoregressive pretraining:

```bash
python data/openwebtext/prepare.py
```

This will generate:

```text
data/openwebtext/train.bin
data/openwebtext/val.bin
```

---

### 3. Pretrain GPT-2 Model

Start pretraining:

```bash
python -u src/train.py
```

For multi-GPU distributed training (DDP):

```bash
torchrun --nproc_per_node=<num_gpus> src/train.py
```

---

### 4. Prepare SFT Data

Prepare instruction tuning data:

```bash
python data/sft_af_opt/Dolly.py
```

This will generate:

```text
data/sft_af_opt/train_sft_dolly.pt
data/sft_af_opt/val_sft_dolly.pt
```

---

### 5. Run Supervised Fine-Tuning (SFT)

Start SFT from the pretrained checkpoint:

```bash
python -u src/sft.py
```

---

### 6. Generate Text

Run inference:

```bash
python src/generate.py
```

---

## One-line Summary

**A research-oriented, from-scratch GPT-2 (~123.65M) implementation inspired by NanoGPT, covering model implementation, pretraining, SFT, and distributed training.**

---
