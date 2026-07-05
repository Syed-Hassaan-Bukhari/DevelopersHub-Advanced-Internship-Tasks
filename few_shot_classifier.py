"""
few_shot_classifier.py
------------------------
Few-shot ticket classification using an instruction-tuned generative LLM
(google/flan-t5-base by default) with in-context examples.

Why not just call `.generate()` and read the text back?
Free-form generation ("Category: Billing and Payments") is brittle to
parse (typos, paraphrased category names) and gives no confidence score,
which we need for the "top-3 probable categories" requirement. Instead
we use the standard few-shot LLM SCORING technique (as used in the GPT-3
paper and HELM-style evaluation harnesses):

  1. Build a single prompt containing a handful of labeled example
     tickets (the few-shot "shots") followed by the new ticket.
  2. For every candidate category, force-decode that category's tokens
     as the continuation and read off the model's log-likelihood of
     producing exactly that string.
  3. Length-normalize the log-likelihood (avg log-prob per token) so
     multi-word categories aren't unfairly penalized versus short ones.
  4. Softmax the normalized scores across candidates to get a ranked,
     comparable probability distribution -> top-3 categories.

This gives few-shot classification a genuine confidence ranking that is
directly comparable to the zero-shot pipeline's output format.
"""

import random
from functools import lru_cache
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import FEW_SHOT_MODEL, CATEGORIES, FEW_SHOT_EXAMPLES, RANDOM_SEED

random.seed(RANDOM_SEED)


@lru_cache(maxsize=1)
def get_few_shot_model(model_name: str = FEW_SHOT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def build_few_shot_prompt(ticket_text: str, shots_per_category: int = 1,
                           categories: List[str] = None) -> str:
    """Builds a prompt with a few labeled examples per category, then the
    new ticket to classify."""
    categories = categories or CATEGORIES
    lines = [
        "Classify each customer support ticket into exactly one category.",
        f"Possible categories: {', '.join(categories)}.",
        "",
    ]
    shot_pool = []
    for cat in categories:
        examples = FEW_SHOT_EXAMPLES.get(cat, [])[:shots_per_category]
        for ex in examples:
            shot_pool.append((ex, cat))
    random.shuffle(shot_pool)

    for ex_text, ex_cat in shot_pool:
        lines.append(f"Ticket: {ex_text}")
        lines.append(f"Category: {ex_cat}")
        lines.append("")

    lines.append(f"Ticket: {ticket_text}")
    lines.append("Category:")
    return "\n".join(lines)


def _score_label(prompt: str, label: str, tokenizer, model) -> float:
    """Length-normalized log-likelihood of `label` as the decoder output
    for the given encoder `prompt`."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    labels = tokenizer(f" {label}", return_tensors="pt").input_ids

    with torch.no_grad():
        out = model(input_ids=inputs.input_ids,
                     attention_mask=inputs.attention_mask,
                     labels=labels)
        # out.loss is the mean per-token cross-entropy (negative log-likelihood)
        avg_nll = out.loss.item()

    return -avg_nll  # higher = more likely


def classify_ticket(text: str, candidate_labels: List[str] = None,
                     shots_per_category: int = 1, top_k: int = 3) -> List[Dict]:
    """Classify a single ticket with few-shot LLM scoring."""
    candidate_labels = candidate_labels or CATEGORIES
    tokenizer, model = get_few_shot_model()
    prompt = build_few_shot_prompt(text, shots_per_category, candidate_labels)

    scores = {}
    for label in candidate_labels:
        scores[label] = _score_label(prompt, label, tokenizer, model)

    # Softmax over avg log-likelihood scores -> comparable probabilities
    logits = torch.tensor(list(scores.values()))
    probs = torch.softmax(logits, dim=0).tolist()

    ranked = sorted(
        [{"label": lbl, "score": p} for lbl, p in zip(scores.keys(), probs)],
        key=lambda d: d["score"], reverse=True,
    )
    return ranked[:top_k]


def classify_batch(texts: List[str], candidate_labels: List[str] = None,
                    shots_per_category: int = 1, top_k: int = 3) -> List[List[Dict]]:
    return [classify_ticket(t, candidate_labels, shots_per_category, top_k) for t in texts]


if __name__ == "__main__":
    sample = "My package tracking hasn't updated in 5 days and it was supposed to arrive already."
    for item in classify_ticket(sample):
        print(f"{item['label']:30s} {item['score']:.3f}")
