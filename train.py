import torch
from torch.utils.data import DataLoader
from model import GPTModel
from tokenizer import get_tokenizer
from config import DEVICE, GPT_CONFIG

def calculate_loss(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), 
                                                   targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), 
                                               targets.view(-1))
        loss.backward()
        optimizer.step()
