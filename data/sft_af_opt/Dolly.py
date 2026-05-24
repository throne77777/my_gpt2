import json
import random
import os
import requests
import tiktoken
import torch
from dataset import *

random.seed(42)

enc = tiktoken.get_encoding("gpt2")

# Dolly 15k 官方 jsonl 文件
url = "https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl"
download_path = os.path.join(os.path.dirname(__file__), "dolly_15k.jsonl")

system_prompt = "You are a helpful assistant."

if not os.path.exists(download_path):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(download_path, "w", encoding="utf-8") as f:
        f.write(r.text)

data = []
with open(download_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            data.append(json.loads(line))

# subset = random.sample(data, 5000)

def dolly_to_messages(ex, system_prompt=None):
    instruction = (ex.get("instruction") or "").strip()
    context = (ex.get("context") or "").strip()
    response = (ex.get("response") or "").strip()

    if context:
        user_content = f"{instruction}\n\n{context}"
    else:
        user_content = instruction

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": response})
    return messages

dataset = []
for item in data:
    msg = dolly_to_messages(item, system_prompt)
    input_ids, labels = tokenize_sft_example(enc, msg, 256)
    dataset.append({"input_ids": input_ids, "labels": labels})

random.shuffle(dataset)

split = int(len(dataset) * 0.9)
train_dataset = dataset[:split]
val_dataset = dataset[split:]

train_path = os.path.join(os.path.dirname(__file__), "train_sft_dolly.pt")
val_path = os.path.join(os.path.dirname(__file__), "val_sft_dolly.pt")

torch.save(train_dataset, train_path)
torch.save(val_dataset, val_path)

print(f"saved train to: {train_path}")
print(f"saved val to:   {val_path}")
print(f"num samples:    {len(dataset)}")