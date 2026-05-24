import torch

IGNORE_INDEX = -100
def apply_chat_template(messages, add_generation_prompt=False):
    ROLE_TOKENS={
        "system": "system:",
        "user": "user:",
        "assistant": "assistant:"
    }

    text = ""

    for msg in messages:

        role = msg["role"]
        content = msg["content"].strip()

        if role not in ROLE_TOKENS:
            raise ValueError(f"Unknown role: {role}")

        text += ROLE_TOKENS[role] + "\n"
        text += content + "\n\n"

    if add_generation_prompt:
        if not text.endswith("assistant:\n"):
            text += "assistant:\n"

    return text

def tokenize_sft_example(tokenizer, messages, block_size):
    input_ids = []
    labels = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"].strip()

        role_text = f"{role}:\n"
        role_ids = tokenizer.encode(role_text)

        content_text = content + "\n\n"
        content_ids = tokenizer.encode(content_text)

        input_ids.extend(role_ids)
        input_ids.extend(content_ids)

        if role == "assistant":
            labels.extend([IGNORE_INDEX] * len(role_ids))
            labels.extend(content_ids)

        else:
            labels.extend([IGNORE_INDEX] * (len(role_ids) + len(content_ids)))

    input_ids = input_ids[:block_size]
    labels = labels[:block_size]

    pad_id = getattr(tokenizer, "eot_token", 50256)

    has_sup = any(l != IGNORE_INDEX for l in labels)

    if len(input_ids) > 0 and has_sup and input_ids[-1] != pad_id and labels[-1] != IGNORE_INDEX:
            input_ids[-1] = pad_id
            labels[-1] = pad_id
    #防止截断的时候把终止符删掉了，导致模型会不断输出垃圾内容
    if len(input_ids) < block_size:
        pad_len = block_size - len(input_ids)

        input_ids.extend([pad_id] * pad_len)
        labels.extend([IGNORE_INDEX] * pad_len)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return input_ids, labels
    #tokenization function is missing


