from train import create_dataloader, train_epoch
from lora import apply_lora
from evaluate import run_mmlu_evaluation
from score import ResponseEvaluator

model = GPTModel(GPT_CONFIG)
model = apply_lora(model)

train_loader = create_dataloader(train_data)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for epoch in range(3):
    train_loss = train_epoch(model, train_loader, optimizer)

run_mmlu_evaluation("out/finetune/lora/final")

evaluator = ResponseEvaluator()
scores = evaluator.score_responses(test_data)
print(f"Average Score: {sum(scores)/len(scores):.2f}")
