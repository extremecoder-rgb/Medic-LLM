from typing import List, Dict
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge1_p": scores["rouge1"].precision,
        "rouge1_r": scores["rouge1"].recall,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rouge2_p": scores["rouge2"].precision,
        "rouge2_r": scores["rouge2"].recall,
        "rougeL_f": scores["rougeL"].fmeasure,
        "rougeL_p": scores["rougeL"].precision,
        "rougeL_r": scores["rougeL"].recall,
    }


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score."""
    smoothie = SmoothingFunction().method4
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)


def compute_exact_match(reference: str, hypothesis: str) -> float:
    """Compute exact match score (0 or 1)."""
    return 1.0 if reference.strip().lower() == hypothesis.strip().lower() else 0.0


def evaluate_predictions(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, float]:
    """Compute all metrics for a set of predictions."""
    scores = {
        "rouge1_f": 0.0,
        "rouge2_f": 0.0,
        "rougeL_f": 0.0,
        "bleu": 0.0,
        "exact_match": 0.0,
    }

    n = len(references)
    for ref, hyp in zip(references, hypotheses):
        rouge = compute_rouge(ref, hyp)
        scores["rouge1_f"] += rouge["rouge1_f"]
        scores["rouge2_f"] += rouge["rouge2_f"]
        scores["rougeL_f"] += rouge["rougeL_f"]
        scores["bleu"] += compute_bleu(ref, hyp)
        scores["exact_match"] += compute_exact_match(ref, hyp)

    for key in scores:
        scores[key] = round(scores[key] / n, 4)

    return scores


def print_evaluation_report(scores: Dict[str, float]):
    """Print formatted evaluation report."""
    print("=" * 50)
    print("Evaluation Report")
    print("=" * 50)
    print(f"ROUGE-1 F1:    {scores['rouge1_f']:.4f}")
    print(f"ROUGE-2 F1:    {scores['rouge2_f']:.4f}")
    print(f"ROUGE-L F1:    {scores['rougeL_f']:.4f}")
    print(f"BLEU:          {scores['bleu']:.4f}")
    print(f"Exact Match:   {scores['exact_match']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    sample_refs = [
        "Metformin is first-line treatment for Type 2 Diabetes",
        "ACE inhibitors are preferred for hypertension in diabetes",
    ]
    sample_hyp = [
        "Metformin is the first line treatment for diabetes",
        "ACE inhibitors are recommended for hypertension",
    ]

    scores = evaluate_predictions(sample_refs, sample_hyp)
    print_evaluation_report(scores)
