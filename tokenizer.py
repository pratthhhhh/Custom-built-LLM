import re
import torch
import tiktoken
from config import DEVICE


class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[s] for s in preprocessed]

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r'\s+([,.?!"()\'])', r'\1', text)


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def format_instruction(entry: dict) -> str:
    instruction = entry.get("instruction", "")
    input_text = entry.get("input", "").strip()
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}"
    return f"### Instruction:\n{instruction}"


def text_to_tensor(text: str, tokenizer) -> torch.Tensor:
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(encoded).unsqueeze(0).to(DEVICE)
