import subprocess
from typing import List, Dict
from tqdm import tqdm
from litgpt import LLM
from tokenizer import format_instruction
from config import DEVICE

def run_mmlu_evaluation(model_path: str, tasks: str = "mmlu_philosophy"):
    command = [
        "litgpt", "evaluate", model_path,
        "--tasks", tasks,
        "--batch_size", "4"
    ]
    subprocess.run(command, check=True)

def generate_responses(model: LLM, dataset: List[Dict]) -> List[str]:
    responses = []
    progress = tqdm(dataset, desc="Generating Responses")
    
    for entry in progress:
        prompt = format_instruction(entry)
        response = model.generate(
            prompt, 
            max_new_tokens=256,
            temperature=0.7
        )
        responses.append(response.strip())
    
    return responses
