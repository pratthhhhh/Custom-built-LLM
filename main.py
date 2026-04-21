import os
import argparse
import torch
from datasets import load_dataset
from model import GPTModel
from config import GPT_CONFIG, DEVICE
from train import create_dataloader, train_epoch, evaluate_model
from lora import apply_lora
from evaluate import run_mmlu_evaluation, generate_responses
from score import ResponseEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="Train and evaluate a GPT model with LoRA")

    # Training
    p.add_argument("--epochs",      type=int,   default=3,    help="Number of training epochs")
    p.add_argument("--batch_size",  type=int,   default=8,    help="Batch size")
    p.add_argument("--lr",          type=float, default=1e-4, help="Learning rate")
    p.add_argument("--test_split",  type=float, default=0.1,  help="Fraction of data for eval (0–1)")

    # LoRA
    p.add_argument("--lora_rank",   type=int,   default=8,    help="LoRA rank")
    p.add_argument("--lora_alpha",  type=float, default=16.0, help="LoRA alpha scaling")

    # Generation / eval
    p.add_argument("--max_new_tokens", type=int,   default=256,  help="Tokens to generate per response")
    p.add_argument("--temperature",    type=float, default=0.7,  help="Sampling temperature")
    p.add_argument("--top_k",          type=int,   default=50,   help="Top-k sampling")

    # Modes
    p.add_argument("--skip_train", action="store_true", help="Skip training, load checkpoint instead")
    p.add_argument("--skip_eval",  action="store_true", help="Skip MMLU + scoring evaluation")
    p.add_argument("--checkpoint", type=str, default="out/finetune/lora/final",
                   help="Path to save/load model checkpoint")
    p.add_argument("--prompt",     type=str, default=None,
                   help="Run a single inference prompt and exit")

    return p.parse_args()


def main():
    args = parse_args()

    # ── Single inference mode ─────────────────────────────────────────────────
    if args.prompt:
        from tokenizer import get_tokenizer, text_to_tensor
        model = GPTModel(GPT_CONFIG).to(DEVICE)
        model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        model.eval()
        tokenizer = get_tokenizer()
        idx = text_to_tensor(args.prompt, tokenizer)
        out = model.generate(idx, args.max_new_tokens, args.temperature, args.top_k)
        new_tokens = out[0][idx.shape[1]:].tolist()
        print(tokenizer.decode(new_tokens))
        return

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("Loading dataset...")
    dataset    = load_dataset("tatsu-lab/alpaca")
    splits     = dataset["train"].train_test_split(test_size=args.test_split, seed=42)
    train_data = [dict(row) for row in splits["train"]]
    test_data  = [dict(row) for row in splits["test"]]
    print(f"  train: {len(train_data):,}  |  eval: {len(test_data):,}")

    # ── Build model ───────────────────────────────────────────────────────────
    model     = GPTModel(GPT_CONFIG).to(DEVICE)
    model     = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  params: {total/1e6:.1f}M total | {trainable/1e6:.2f}M trainable (LoRA)")

    # ── Training ──────────────────────────────────────────────────────────────
    if not args.skip_train:
        train_loader = create_dataloader(train_data, batch_size=args.batch_size)
        eval_loader  = create_dataloader(test_data,  batch_size=args.batch_size)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, train_loader, optimizer)
            eval_loss  = evaluate_model(model, eval_loader)
            print(f"Epoch {epoch+1}/{args.epochs} | train: {train_loss:.4f} | eval: {eval_loss:.4f}")
        os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
        print(f"Saved checkpoint -> {args.checkpoint}")
    else:
        checkpoint = torch.load(args.checkpoint, map_location=DEVICE)
        model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint <- {args.checkpoint}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    if not args.skip_eval:
        run_mmlu_evaluation(model)
        responses   = generate_responses(model, test_data[:50],
                                         args.max_new_tokens, args.temperature, args.top_k)
        scored_data = [{**entry, "response": r}
                       for entry, r in zip(test_data[:50], responses)]
        evaluator = ResponseEvaluator()
        scores    = evaluator.score_responses(scored_data)
        print(f"Average ROUGE-L Score: {sum(scores)/len(scores):.2f}")


if __name__ == "__main__":
    main()