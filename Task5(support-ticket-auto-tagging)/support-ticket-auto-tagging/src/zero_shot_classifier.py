"""
zero_shot_classifier.py
------------------------
Zero-shot ticket classification using HuggingFace's
`zero-shot-classification` pipeline (NLI-based, facebook/bart-large-mnli
by default).

How it works:
The NLI model is asked, for every candidate category, whether the
ticket text ENTAILS the hypothesis "This support ticket is about
<category>." The entailment probabilities across all categories are
normalized (multi_label=False) into a single probability distribution,
which we treat directly as the model's category ranking -- no
task-specific training data required.
"""

from functools import lru_cache
from typing import List, Dict

from transformers import pipeline

from src.config import ZERO_SHOT_MODEL, ZERO_SHOT_HYPOTHESIS_TEMPLATE, CATEGORIES


@lru_cache(maxsize=1)
def get_zero_shot_pipeline(model_name: str = ZERO_SHOT_MODEL):
    """Cached pipeline loader so the model is only downloaded/loaded once."""
    return pipeline("zero-shot-classification", model=model_name)


def classify_ticket(text: str, candidate_labels: List[str] = None,
                     top_k: int = 3) -> List[Dict]:
    """
    Classify a single ticket with zero-shot NLI.

    Returns a list of {"label": str, "score": float} sorted descending,
    truncated to top_k.
    """
    candidate_labels = candidate_labels or CATEGORIES
    clf = get_zero_shot_pipeline()
    result = clf(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template=ZERO_SHOT_HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    ranked = [{"label": lbl, "score": float(score)}
              for lbl, score in zip(result["labels"], result["scores"])]
    return ranked[:top_k]


def classify_batch(texts: List[str], candidate_labels: List[str] = None,
                    top_k: int = 3) -> List[List[Dict]]:
    """Classify many tickets; simple loop wrapper (pipeline batches internally
    when given a list, but we keep results grouped per-ticket for clarity)."""
    candidate_labels = candidate_labels or CATEGORIES
    clf = get_zero_shot_pipeline()
    results = clf(
        texts,
        candidate_labels=candidate_labels,
        hypothesis_template=ZERO_SHOT_HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )
    # pipeline returns a single dict if given a single string; normalize to list
    if isinstance(results, dict):
        results = [results]

    all_ranked = []
    for result in results:
        ranked = [{"label": lbl, "score": float(score)}
                  for lbl, score in zip(result["labels"], result["scores"])]
        all_ranked.append(ranked[:top_k])
    return all_ranked


if __name__ == "__main__":
    sample = "My package tracking hasn't updated in 5 days and it was supposed to arrive already."
    for item in classify_ticket(sample):
        print(f"{item['label']:30s} {item['score']:.3f}")
