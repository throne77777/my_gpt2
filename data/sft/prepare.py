import json
import random
import os
import requests
import tiktoken
import torch
from dataset import *

random.seed(42)

enc = tiktoken.get_encoding('gpt2')
url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
download_path = os.path.join(os.path.dirname(__file__),'input.json')
output_path = os.path.join(os.path.dirname(__file__),'output.json')
system_prompt = "You are a helpful assistant."

if not os.path.exists(download_path):
    with open(download_path,'w', encoding="utf-8") as f:
        f.write(requests.get(url).text)

with open(download_path,"r",encoding="utf-8") as f:
    data = json.load(f)
#return a list of dic with each dic contains a whole message

subset = random.sample(data, 5000)

def alpaca_to_messages(ex, system_prompt=None):
    instruction = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    if inp:
        user_content = f"{instruction}\n\n{inp}"
    else:
        user_content = instruction

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": out})
    return messages

dataset=[]
for item in subset:
    msg = alpaca_to_messages(item,system_prompt)
    input_ids,labels=tokenize_sft_example(enc,msg, 256)
    dataset.append({"input_ids":input_ids,"labels":labels})
# print(dataset[0]["input_ids"].shape, dataset[0]["labels"].shape)
# print((dataset[0]["labels"] != -100).sum().item())
random.shuffle(dataset)

split = int(len(dataset) * 0.9)

train_dataset = dataset[:split]
val_dataset   = dataset[split:]

train_path = os.path.join(os.path.dirname(__file__), "train_sft.pt")
val_path   = os.path.join(os.path.dirname(__file__), "val_sft.pt")

torch.save(train_dataset, train_path)
torch.save(val_dataset, val_path)