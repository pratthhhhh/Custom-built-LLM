import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict
from tqdm import tqdm
from config import DEVICE, GPT_CONFIG
from model import GPTModel
from tokenizer import get_tokenizer

class InstructionDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_data = []
        
        for entry in data:
            formatted = format_instruction(entry) + f"\n\n### Response:\n{entry['output']}"
            encoded = tokenizer.encode(formatted, allowed_special={'<|endoftext|>'})
            self.encoded_data.append(torch.tensor(encoded[:self.max_length]))

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        item = self.encoded_data[idx]
        x = item[:-1]
        y = item[1:]
        return x, y

def create_dataloader(data: List[Dict], batch_size: int = 8) -> DataLoader:
    tokenizer = get_tokenizer()
    dataset = InstructionDataset(data, tokenizer, GPT_CONFIG["context_length"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_epoch(model: GPTModel, train_loader: DataLoader, optimizer: torch.optim.Optimizer):
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc="Training")
    
    for inputs, targets in progress:
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            targets.view(-1),
            ignore_index=-100
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(train_loader)

def evaluate_model(model: GPTModel, eval_loader: DataLoader):
    model.eval()
    total_loss = 0
    progress = tqdm(eval_loader, desc="Evaluating")
    
    with torch.no_grad():
        for inputs, targets in progress:
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-100
            )
            total_loss += loss.item()
            progress.set_postfix({"eval_loss": f"{loss.item():.4f}"})
    
    return total_loss / len(eval_loader)
