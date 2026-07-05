"""
evaluate.py
------------
Loads the exported joblib pipeline(s) and the held-out test split saved
by src/train.py, and computes accuracy, precision, recall, F1, ROC-AUC,
confusion matrix, and a full classification report.

Usage:
    python -m src.evaluate
"""

import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve,
)

from src.config import MODELS_DIR, RESULTS_DIR, PARAM_GRIDS


def load_test_split():
    X_test = pd.read_csv(RESULTS_DIR / "X_test.csv")
    y_test = pd.read_csv(RESULTS_DIR / "y_test.csv").iloc[:, 0]
    return X_test, y_test


def evaluate_pipeline(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "y_pred": y_pred.tolist(),
        "y_proba": y_proba.tolist(),
    }


def main():
    X_test, y_test = load_test_split()

    all_metrics = {}
    for model_name in PARAM_GRIDS:
        model_path = MODELS_DIR / f"{model_name}_pipeline.joblib"
        if not model_path.exists():
            print(f"Skipping {model_name}: {model_path} not found. Run `python -m src.train` first.")
            continue
        pipeline = joblib.load(model_path)
        metrics = evaluate_pipeline(pipeline, X_test, y_test)
        all_metrics[model_name] = metrics

        print(f"\n=== {model_name} ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(metrics["classification_report"])

    # Save everything except the bulky raw prediction arrays in the summary JSON
    summary = {
        name: {k: v for k, v in m.items() if k not in ("y_pred", "y_proba", "roc_curve")}
        for name, m in all_metrics.items()
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save predictions for the winning pipeline for inspection / Streamlit
    with open(RESULTS_DIR / "full_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nSaved metrics to {RESULTS_DIR / 'metrics.json'}")
    return all_metrics


if __name__ == "__main__":
    main()
