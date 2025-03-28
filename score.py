from typing import List, Dict
from tqdm import tqdm
from litgpt import LLM
from config import DEVICE

class ResponseEvaluator:
    def __init__(self, judge_model: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.judge = LLM.load(judge_model)
        self.judge.eval().to(DEVICE)

    def _create_prompt(self, instruction: str, reference: str, response: str) -> str:
        return f"""Instruction: {instruction}
        Correct Answer: {reference}
        Model Response: {response}
        
        Please score the model response's accuracy and completeness on a scale from 0 to 100.
        Respond only with the numerical score, no other text.
        Score: """

    def score_response(self, instruction: str, reference: str, response: str) -> int:
        prompt = self._create_prompt(instruction, reference, response)
        score_text = self.judge.generate(prompt, max_new_tokens=10)
        
        try:
            return int(score_text.strip())
        except ValueError:
            return -1  # Indicate invalid score

    def score_responses(self, dataset: List[Dict], response_key: str = "response") -> List[int]:
        scores = []
        valid_responses = 0
        
        progress = tqdm(dataset, desc="Evaluating Responses")
        for entry in progress:
            score = self.score_response(
                entry["instruction"],
                entry["output"],
                entry[response_key]
            )
            
            if score != -1:
                scores.append(score)
                valid_responses += 1
                
            progress.set_postfix({
                "valid": valid_responses,
                "current_score": f"{score if score != -1 else 'invalid'}"
            })
        
        return scores
