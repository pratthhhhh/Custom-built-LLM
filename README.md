# Custom-built LLM

A 124M parameter GPT-based language model built from scratch in PyTorch, featuring multi-head self-attention with rotary positional embeddings (RoPE), LoRA fine-tuning, and a full training/evaluation pipeline on the Alpaca instruction dataset.

## Architecture

| Hyperparameter | Value |
|---|---|
| Parameters | ~124M (with weight tying) |
| Embedding dimension | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Context length | 1024 tokens |
| Vocabulary size | 50,257 (GPT-2 BPE) |
| Positional encoding | Rotary (RoPE) |
| Fine-tuning | LoRA (rank 8–16) |

## Project Structure

| File | Description |
|---|---|
| `main.py` | Entry point — training, evaluation, and inference with CLI flags |
| `model.py` | GPT model: RoPE, multi-head attention, feed-forward, transformer blocks |
| `config.py` | Model hyperparameters |
| `tokenizer.py` | GPT-2 BPE tokenizer wrapper + instruction formatter |
| `train.py` | Training loop, dataset, dataloader with padding collation |
| `lora.py` | LoRA layer implementation and model patching |
| `evaluate.py` | Response generation and MMLU evaluation |
| `score.py` | ROUGE-L scoring of generated responses |

## Setup

```bash
git clone https://github.com/pratthhhhh/Custom-built-LLM.git
cd Custom-built-LLM
pip install -r requirements.txt
```

## Usage

### Train
```bash
python main.py --epochs 3 --batch_size 16 --lr 2e-4 --lora_rank 16
```

### Inference
```bash
python main.py --skip_train --skip_eval --lora_rank 16 --prompt "What is machine learning?"
```

### Evaluate only
```bash
python main.py --skip_train --lora_rank 16
```

### All CLI flags

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 3 | Training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--lora_rank` | 8 | LoRA rank |
| `--lora_alpha` | 16 | LoRA alpha |
| `--temperature` | 0.7 | Sampling temperature |
| `--top_k` | 50 | Top-k sampling |
| `--test_split` | 0.1 | Eval data fraction |
| `--checkpoint` | `out/finetune/lora/final` | Save/load path |
| `--skip_train` | — | Skip to eval/inference |
| `--skip_eval` | — | Skip MMLU scoring |
| `--prompt` | — | Single inference prompt |

## Dataset

Fine-tuned on [tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) — 52K instruction-following examples in `instruction / input / output` format, with a 90/10 train/eval split.

## Notebook

`test.ipynb` provides an interactive inference demo — load the checkpoint and run prompts cell by cell.