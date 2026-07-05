"""
visualize.py
-------------
Generates evaluation charts for the fine-tuned BERT news classifier:
confusion matrix, per-class F1, and confidence distribution.

Run after src/evaluate.py has produced results/metrics.json and
results/predictions.csv.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

from src.config import LABEL_NAMES, RESULTS_DIR, FIGURES_DIR

sns.set_theme(style="whitegrid")


def load_results():
    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    predictions_df = pd.read_csv(RESULTS_DIR / "predictions.csv")
    return metrics, predictions_df


def plot_confusion_matrix(metrics, save_path=None):
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("BERT News Topic Classifier — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path or FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_per_class_metrics(predictions_df, save_path=None):
    from sklearn.metrics import precision_score, recall_score

    p = precision_score(predictions_df["true_label"], predictions_df["predicted_label"],
                         labels=LABEL_NAMES, average=None, zero_division=0)
    r = recall_score(predictions_df["true_label"], predictions_df["predicted_label"],
                      labels=LABEL_NAMES, average=None, zero_division=0)
    f1 = f1_score(predictions_df["true_label"], predictions_df["predicted_label"],
                   labels=LABEL_NAMES, average=None, zero_division=0)

    x = np.arange(len(LABEL_NAMES))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, p, width, label="Precision", color="#4C72B0")
    ax.bar(x, r, width, label="Recall", color="#DD8452")
    ax.bar(x + width, f1, width, label="F1-score", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-class Precision / Recall / F1")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path or FIGURES_DIR / "per_class_metrics.png", dpi=150)
    plt.close()


def plot_confidence_distribution(predictions_df, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=predictions_df, x="confidence", hue="correct", bins=30,
                 element="step", stat="density", common_norm=False, ax=ax,
                 palette={True: "#55A868", False: "#C44E52"})
    ax.set_title("Prediction Confidence: Correct vs Incorrect")
    ax.set_xlabel("Model confidence (softmax probability)")
    plt.tight_layout()
    plt.savefig(save_path or FIGURES_DIR / "confidence_distribution.png", dpi=150)
    plt.close()


def plot_training_history(log_history, save_path=None):
    """Optional: plot train/eval loss curves if a Trainer log_history list
    (trainer.state.log_history) is passed in, e.g. from the notebook."""
    epochs, train_loss, eval_loss, eval_f1 = [], [], [], []
    for entry in log_history:
        if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:
            train_loss.append((entry["epoch"], entry["loss"]))
        if "eval_loss" in entry:
            eval_loss.append((entry["epoch"], entry["eval_loss"]))
        if "eval_f1_macro" in entry:
            eval_f1.append((entry["epoch"], entry["eval_f1_macro"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    if train_loss:
        ax1.plot(*zip(*train_loss), label="Train loss", marker="o")
    if eval_loss:
        ax1.plot(*zip(*eval_loss), label="Eval loss", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training / Evaluation Loss")
    ax1.legend()

    if eval_f1:
        ax2.plot(*zip(*eval_f1), label="Eval F1 (macro)", marker="o", color="#55A868")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 (macro)")
    ax2.set_title("Evaluation F1 across epochs")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path or FIGURES_DIR / "training_curves.png", dpi=150)
    plt.close()


def generate_all_plots():
    metrics, predictions_df = load_results()
    plot_confusion_matrix(metrics)
    plot_per_class_metrics(predictions_df)
    plot_confidence_distribution(predictions_df)
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_plots()
