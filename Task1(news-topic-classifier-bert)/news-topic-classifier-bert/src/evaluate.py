"""
evaluate.py
------------
Loads the fine-tuned model from saved_model/ and produces a full
evaluation report on the AG News test set: accuracy, per-class
precision/recall/F1, confusion matrix, and misclassified examples.

Usage:
    python -m src.evaluate
"""

import json
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import MODEL_DIR, LABEL_NAMES, RESULTS_DIR, MAX_LENGTH
from src.data_utils import load_ag_news


def load_finetuned_model(model_dir=MODEL_DIR):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def predict_batch(texts, tokenizer, model, batch_size: int = 32):
    """Returns (pred_label_ids, pred_probs) for a list of raw texts."""
    all_preds, all_probs = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())
    return all_preds, all_probs


def run_evaluation(n_samples: int = None):
    raw = load_ag_news()
    test = raw["test"]
    if n_samples:
        test = test.shuffle(seed=42).select(range(min(n_samples, len(test))))

    texts = test["text"]
    true_ids = test["label"]

    tokenizer, model = load_finetuned_model()
    pred_ids, pred_probs = predict_batch(texts, tokenizer, model)

    true_labels = [LABEL_NAMES[i] for i in true_ids]
    pred_labels = [LABEL_NAMES[i] for i in pred_ids]

    acc = accuracy_score(true_labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, labels=LABEL_NAMES, average="macro", zero_division=0
    )
    cm = confusion_matrix(true_labels, pred_labels, labels=LABEL_NAMES)
    report = classification_report(true_labels, pred_labels, labels=LABEL_NAMES, zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "labels": LABEL_NAMES,
        "classification_report": report,
    }

    predictions_df = pd.DataFrame({
        "text": texts,
        "true_label": true_labels,
        "predicted_label": pred_labels,
        "confidence": [max(p) for p in pred_probs],
        "correct": [t == p for t, p in zip(true_labels, pred_labels)],
    })

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    predictions_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy: {acc:.4f} | Precision: {precision:.4f} | "
          f"Recall: {recall:.4f} | F1 (macro): {f1:.4f}")
    print("\n" + report)

    return metrics, predictions_df


if __name__ == "__main__":
    run_evaluation()
