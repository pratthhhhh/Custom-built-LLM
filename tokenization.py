import re
import tiktoken

class SimpleTokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!\"()\\']|--|\\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        return [self.str_to_int[s] for s in preprocessed]
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        return re.sub(r"\s+([,.?!\"()\\'])', r'\1", text)

def create_dataloader(text, batch_size=8, seq_length=256, stride=256):
    pass
