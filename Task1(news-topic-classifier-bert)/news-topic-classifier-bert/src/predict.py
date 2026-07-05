"""
predict.py
-----------
Small inference helper: load the fine-tuned model once, classify a
single headline/article, return label + confidence for all 4 classes.
Used by the Streamlit app and importable from the notebook.
"""

from functools import lru_cache
import torch

from src.config import MODEL_DIR, LABEL_NAMES, MAX_LENGTH
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@lru_cache(maxsize=1)
def _load():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict(text: str):
    """Returns a list of {"label": str, "score": float} for all 4 classes,
    sorted by score descending."""
    tokenizer, model = _load()
    inputs = tokenizer(text, truncation=True, padding=True,
                        max_length=MAX_LENGTH, return_tensors="pt")
    logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    ranked = sorted(
        [{"label": lbl, "score": p} for lbl, p in zip(LABEL_NAMES, probs)],
        key=lambda d: d["score"], reverse=True,
    )
    return ranked


if __name__ == "__main__":
    sample = "The central bank raised interest rates for the third time this year."
    for item in predict(sample):
        print(f"{item['label']:10s} {item['score']:.3f}")
