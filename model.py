def load_pretrained_weights(model, pretrained_path):
    pass

GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1
}

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPTModel(GPT_CONFIG).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004)

    for epoch in range(10):
        train_epoch(model, train_loader, optimizer, device)
        val_loss = calculate_loss(model, val_loader, device)
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.3f}")
