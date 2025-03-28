import torch.nn as nn
from torch import Tensor
from config import DEVICE

class LoRALayer(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original parameters
        for param in self.original.parameters():
            param.requires_grad = False
            
        # Initialize LoRA matrices
        self.lora_A = nn.Linear(
            original_layer.in_features, 
            rank, 
            bias=False
        ).to(DEVICE)
        
        self.lora_B = nn.Linear(
            rank, 
            original_layer.out_features, 
            bias=False
        ).to(DEVICE)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(x))
        return original_out + (self.alpha / self.rank) * lora_out

def apply_lora(model: nn.Module, rank: int = 8):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Linear):
            setattr(model, name, LoRALayer(layer, rank))
        else:
            apply_lora(layer, rank)
    return model.to(DEVICE)
