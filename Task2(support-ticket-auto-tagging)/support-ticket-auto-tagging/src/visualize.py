"""
visualize.py
-------------
Generates all comparison plots for the zero-shot vs few-shot ticket
classifiers and saves them under results/figures/.

Run after src/evaluate.py has produced results/metrics.json and
results/predictions.csv.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import CATEGORIES, RESULTS_DIR, FIGURES_DIR

sns.set_theme(style="whitegrid")


def load_results():
    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    predictions_df = pd.read_csv(RESULTS_DIR / "predictions.csv")
    return metrics, predictions_df


def plot_metric_comparison(metrics: dict, save_path=None):
    """Grouped bar chart: accuracy / precision / recall / F1 / top-3 acc."""
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "top3_accuracy"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score", "Top-3 Accuracy"]

    zs_vals = [metrics["zero_shot"][k] for k in metric_keys]
    fs_vals = [metrics["few_shot"][k] for k in metric_keys]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars1 = ax.bar(x - width / 2, zs_vals, width, label="Zero-shot", color="#4C72B0")
    bars2 = ax.bar(x + width / 2, fs_vals, width, label="Few-shot", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title("Zero-shot vs Few-shot: Classification Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend()

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path or FIGURES_DIR / "metric_comparison.png", dpi=150)
    plt.close()


def plot_confusion_matrix(cm, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CATEGORIES,
                yticklabels=CATEGORIES, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_per_category_f1(metrics: dict, predictions_df: pd.DataFrame, save_path=None):
    """Per-category F1 comparison using sklearn classification_report text
    parsed back is fragile -- recompute directly from predictions instead."""
    from sklearn.metrics import f1_score

    zs_f1 = f1_score(predictions_df["true_category"], predictions_df["zero_shot_top1"],
                      labels=CATEGORIES, average=None, zero_division=0)
    fs_f1 = f1_score(predictions_df["true_category"], predictions_df["few_shot_top1"],
                      labels=CATEGORIES, average=None, zero_division=0)

    x = np.arange(len(CATEGORIES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, zs_f1, width, label="Zero-shot", color="#4C72B0")
    ax.bar(x + width / 2, fs_f1, width, label="Few-shot", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES, rotation=35, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_title("Per-category F1-score: Zero-shot vs Few-shot")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path or FIGURES_DIR / "per_category_f1.png", dpi=150)
    plt.close()


def plot_confidence_distribution(predictions_df: pd.DataFrame, results_raw=None, save_path=None):
    """Shows correctness rate stratified by whether prediction was top-1
    correct vs only top-3 correct, for both methods."""
    def bucket(row, method):
        if row[f"{method}_top1"] == row["true_category"]:
            return "Top-1 correct"
        elif row[f"{method}_top3_correct"]:
            return "In top-3, not top-1"
        else:
            return "Missed (not in top-3)"

    zs_buckets = predictions_df.apply(lambda r: bucket(r, "zero_shot"), axis=1).value_counts()
    fs_buckets = predictions_df.apply(lambda r: bucket(r, "few_shot"), axis=1).value_counts()

    order = ["Top-1 correct", "In top-3, not top-1", "Missed (not in top-3)"]
    zs_vals = [zs_buckets.get(o, 0) for o in order]
    fs_vals = [fs_buckets.get(o, 0) for o in order]

    x = np.arange(len(order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, zs_vals, width, label="Zero-shot", color="#4C72B0")
    ax.bar(x + width / 2, fs_vals, width, label="Few-shot", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Number of tickets")
    ax.set_title("Where correct answers land in the ranked predictions")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path or FIGURES_DIR / "ranking_breakdown.png", dpi=150)
    plt.close()


def generate_all_plots():
    metrics, predictions_df = load_results()

    plot_metric_comparison(metrics)
    plot_confusion_matrix(np.array(metrics["zero_shot"]["confusion_matrix"]),
                           "Zero-shot Confusion Matrix", FIGURES_DIR / "confusion_zero_shot.png")
    plot_confusion_matrix(np.array(metrics["few_shot"]["confusion_matrix"]),
                           "Few-shot Confusion Matrix", FIGURES_DIR / "confusion_few_shot.png")
    plot_per_category_f1(metrics, predictions_df)
    plot_confidence_distribution(predictions_df)

    print(f"Saved 5 figures to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_plots()
