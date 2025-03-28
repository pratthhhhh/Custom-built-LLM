import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, n_heads):
        super().__init__()
        pass

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_emb = nn.Embedding(config["context_length"], config["emb_dim"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config["emb_dim"], config["n_heads"]) 
              for _ in range(config["n_layers"])])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"])
        
    def forward(self, inputs):
        pass
