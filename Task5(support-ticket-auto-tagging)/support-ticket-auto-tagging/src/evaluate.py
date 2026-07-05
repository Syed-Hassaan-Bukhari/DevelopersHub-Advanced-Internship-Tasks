"""
evaluate.py
------------
Runs zero-shot and few-shot classifiers over the held-out test split and
computes standard classification metrics, plus a top-3 "hit rate" that
credits a prediction if the true category appears anywhere in the
model's top-3 ranked output (directly reflecting the project's top-3
prediction requirement).
"""

import json
import time
from typing import List, Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report,
)

from src.config import CATEGORIES, RESULTS_DIR
from src.preprocessing import load_dataset, split_dataset
from src import zero_shot_classifier as zsc
from src import few_shot_classifier as fsc


def top1_labels(ranked_results: List[List[Dict]]) -> List[str]:
    return [r[0]["label"] for r in ranked_results]


def top3_hit(true_label: str, ranked_result: List[Dict]) -> bool:
    return true_label in [r["label"] for r in ranked_result]


def compute_metrics(y_true: List[str], y_pred: List[str], labels=CATEGORIES) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def run_full_evaluation(n_test_samples: int = None, shots_per_category: int = 1):
    """
    Runs both classifiers on the test split, computes metrics, and saves:
      - results/predictions.csv       (per-ticket predictions from both methods)
      - results/metrics.json          (accuracy/precision/recall/f1/top-3 hit rate)
    """
    df = load_dataset()
    _, test_df = split_dataset(df)
    if n_test_samples:
        test_df = test_df.sample(n=min(n_test_samples, len(test_df)), random_state=42)

    texts = test_df["text"].tolist()
    true_labels = test_df["category"].tolist()

    print(f"Running zero-shot classification on {len(texts)} tickets...")
    t0 = time.time()
    zero_shot_results = zsc.classify_batch(texts, top_k=3)
    zero_shot_time = time.time() - t0

    print(f"Running few-shot classification on {len(texts)} tickets...")
    t0 = time.time()
    few_shot_results = fsc.classify_batch(texts, shots_per_category=shots_per_category, top_k=3)
    few_shot_time = time.time() - t0

    zs_top1 = top1_labels(zero_shot_results)
    fs_top1 = top1_labels(few_shot_results)

    zs_top3_hits = [top3_hit(t, r) for t, r in zip(true_labels, zero_shot_results)]
    fs_top3_hits = [top3_hit(t, r) for t, r in zip(true_labels, few_shot_results)]

    zs_metrics = compute_metrics(true_labels, zs_top1)
    fs_metrics = compute_metrics(true_labels, fs_top1)
    zs_metrics["top3_accuracy"] = sum(zs_top3_hits) / len(zs_top3_hits)
    fs_metrics["top3_accuracy"] = sum(fs_top3_hits) / len(fs_top3_hits)
    zs_metrics["avg_inference_time_sec"] = zero_shot_time / len(texts)
    fs_metrics["avg_inference_time_sec"] = few_shot_time / len(texts)

    predictions_df = pd.DataFrame({
        "ticket_id": test_df["ticket_id"].values,
        "text": texts,
        "true_category": true_labels,
        "zero_shot_top1": zs_top1,
        "zero_shot_top3": [", ".join(l["label"] for l in r) for r in zero_shot_results],
        "zero_shot_top3_correct": zs_top3_hits,
        "few_shot_top1": fs_top1,
        "few_shot_top3": [", ".join(l["label"] for l in r) for r in few_shot_results],
        "few_shot_top3_correct": fs_top3_hits,
    })

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    predictions_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump({"zero_shot": zs_metrics, "few_shot": fs_metrics}, f, indent=2)

    print("\n=== Zero-shot metrics ===")
    print(f"Accuracy: {zs_metrics['accuracy']:.3f} | "
          f"Precision: {zs_metrics['precision_macro']:.3f} | "
          f"Recall: {zs_metrics['recall_macro']:.3f} | "
          f"F1: {zs_metrics['f1_macro']:.3f} | "
          f"Top-3 Acc: {zs_metrics['top3_accuracy']:.3f}")

    print("\n=== Few-shot metrics ===")
    print(f"Accuracy: {fs_metrics['accuracy']:.3f} | "
          f"Precision: {fs_metrics['precision_macro']:.3f} | "
          f"Recall: {fs_metrics['recall_macro']:.3f} | "
          f"F1: {fs_metrics['f1_macro']:.3f} | "
          f"Top-3 Acc: {fs_metrics['top3_accuracy']:.3f}")

    return zs_metrics, fs_metrics, predictions_df


if __name__ == "__main__":
    run_full_evaluation()
