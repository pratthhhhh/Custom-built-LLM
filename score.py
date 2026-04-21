from typing import List, Dict
from tqdm import tqdm
from rouge_score import rouge_scorer


class ResponseEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def score_response(self, reference: str, response: str) -> int:
        result = self.scorer.score(reference, response)
        # scale rougeL F1 to 0–100
        return round(result["rougeL"].fmeasure * 100)

    def score_responses(self, dataset: List[Dict],
                        response_key: str = "response") -> List[int]:
        scores = []
        for entry in tqdm(dataset, desc="Scoring Responses"):
            score = self.score_response(entry["output"], entry[response_key])
            scores.append(score)
        return scores