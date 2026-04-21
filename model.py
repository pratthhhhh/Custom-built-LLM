import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPT_CONFIG, DEVICE


def precompute_rope(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len)
    freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim//2)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    seq_len = x.shape[-2]
    cos, sin = cos[:seq_len], sin[:seq_len]
    x_even, x_odd = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([x_even * cos - x_odd * sin,
                           x_even * sin + x_odd * cos], dim=-1)
    return rotated.flatten(-2)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, drop_rate: float = 0.1):
        super().__init__()
        assert emb_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.drop_rate = drop_rate
        cos, sin = precompute_rope(self.head_dim, GPT_CONFIG["context_length"])
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, self.cos, self.sin)
        k = apply_rope(k, self.cos, self.sin)
        dropout_p = self.drop_rate if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim: int, n_heads: int, drop_rate: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(emb_dim, n_heads, drop_rate)
        self.ff = FeedForward(emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class GPTModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.drop = nn.Dropout(config["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(config["emb_dim"], config["n_heads"], config["drop_rate"])
              for _ in range(config["n_layers"])])
        self.norm = nn.LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        self.out_head.weight = self.tok_emb.weight  # weight tying -> ~124M params

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.tok_emb(inputs))
        x = self.trf_blocks(x)
        x = self.norm(x)
        return self.out_head(x)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        ctx_len = GPT_CONFIG["context_length"]
        for _ in range(max_new_tokens):
            logits = self(idx[:, -ctx_len:])[:, -1, :]
            if temperature != 1.0:
                logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


def load_pretrained(model: GPTModel, path: str) -> GPTModel:
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model.to(DEVICE)