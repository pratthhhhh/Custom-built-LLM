import torch
from typing import List, Dict
from tqdm import tqdm
from datasets import load_dataset
from tokenizer import get_tokenizer, text_to_tensor, format_instruction
from config import DEVICE


def generate_responses(model, dataset: List[Dict],
                       max_new_tokens: int = 256,
                       temperature: float = 0.7,
                       top_k: int = 50) -> List[str]:
    tokenizer = get_tokenizer()
    model.eval()
    responses = []
    for entry in tqdm(dataset, desc="Generating Responses"):
        prompt = format_instruction(entry)
        idx = text_to_tensor(prompt, tokenizer)
        with torch.no_grad():
            out = model.generate(idx, max_new_tokens, temperature, top_k)
        decoded = tokenizer.decode(out[0].tolist())
        # strip the prompt prefix, keep only the generated response
        responses.append(decoded[len(prompt):].strip())
    return responses


def run_mmlu_evaluation(model, split: str = "test",
                        subject: str = "philosophy",
                        max_samples: int = 100) -> float:
    tokenizer = get_tokenizer()
    dataset = load_dataset("cais/mmlu", subject, split=split)
    choices_labels = ["A", "B", "C", "D"]
    correct = 0
    total   = 0

    model.eval()
    for row in tqdm(dataset.select(range(min(max_samples, len(dataset)))),
                    desc=f"MMLU ({subject})"):
        options = "\n".join(f"{choices_labels[i]}. {row['choices'][i]}"
                            for i in range(len(row['choices'])))
        prompt = (f"Question: {row['question']}\n{options}\n"
                  f"Answer with a single letter (A/B/C/D):")
        idx = text_to_tensor(prompt, tokenizer)
        with torch.no_grad():
            logits = model(idx[:, -1024:])[:, -1, :]
        # pick the highest-probability answer token
        token_ids = [tokenizer.encode(f" {c}")[0] for c in choices_labels]
        scores    = logits[0, token_ids]
        predicted = choices_labels[scores.argmax().item()]
        if predicted == choices_labels[row["answer"]]:
            correct += 1
        total += 1

    accuracy = correct / total if total else 0.0
    print(f"MMLU {subject} accuracy: {correct}/{total} = {accuracy:.2%}")
    return accuracy