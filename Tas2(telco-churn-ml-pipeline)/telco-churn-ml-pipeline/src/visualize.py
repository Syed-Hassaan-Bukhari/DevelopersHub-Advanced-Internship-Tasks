"""
visualize.py
-------------
Generates all comparison charts for the Telco churn pipeline: metric
comparison bar chart, confusion matrices, ROC curves, and feature
importance (Random Forest) / coefficients (Logistic Regression).

Run after src/evaluate.py has produced results/metrics.json and
results/full_metrics.json.
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import RESULTS_DIR, FIGURES_DIR, MODELS_DIR, PARAM_GRIDS

sns.set_theme(style="whitegrid")


def load_results():
    with open(RESULTS_DIR / "metrics.json") as f:
        metrics = json.load(f)
    with open(RESULTS_DIR / "full_metrics.json") as f:
        full_metrics = json.load(f)
    return metrics, full_metrics


def plot_metric_comparison(metrics: dict, save_path=None):
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    labels = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    model_names = list(metrics.keys())

    x = np.arange(len(labels))
    width = 0.8 / len(model_names)
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, model_name in enumerate(model_names):
        vals = [metrics[model_name][k] for k in metric_keys]
        offset = (i - (len(model_names) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=model_name.replace("_", " ").title(),
                       color=colors[i % len(colors)])
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison: Logistic Regression vs Random Forest")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path or FIGURES_DIR / "metric_comparison.png", dpi=150)
    plt.close()


def plot_confusion_matrices(full_metrics: dict, save_path=None):
    n = len(full_metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (model_name, m) in zip(axes, full_metrics.items()):
        cm = np.array(m["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"], ax=ax)
        ax.set_title(model_name.replace("_", " ").title())
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path or FIGURES_DIR / "confusion_matrices.png", dpi=150)
    plt.close()


def plot_roc_curves(full_metrics: dict, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    for model_name, m in full_metrics.items():
        fpr = m["roc_curve"]["fpr"]
        tpr = m["roc_curve"]["tpr"]
        ax.plot(fpr, tpr, label=f"{model_name.replace('_', ' ').title()} (AUC={m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path or FIGURES_DIR / "roc_curves.png", dpi=150)
    plt.close()


def _feature_names_from_preprocessor(preprocessor):
    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_features = list(cat_encoder.get_feature_names_out(preprocessor.transformers_[1][2]))
    return list(num_features) + cat_features


def plot_feature_importance(model_name: str, save_path=None, top_n: int = 15):
    """Works for either model: Random Forest -> feature_importances_,
    Logistic Regression -> absolute standardized coefficients."""
    pipeline = joblib.load(MODELS_DIR / f"{model_name}_pipeline.joblib")
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = _feature_names_from_preprocessor(preprocessor)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        xlabel = "Feature importance (Gini)"
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        xlabel = "|Coefficient| (standardized features)"
    else:
        return

    order = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in order]
    top_values = importances[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_features[::-1], top_values[::-1], color="#4C72B0")
    ax.set_xlabel(xlabel)
    ax.set_title(f"Top {top_n} Features — {model_name.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(save_path or FIGURES_DIR / f"feature_importance_{model_name}.png", dpi=150)
    plt.close()


def generate_all_plots():
    metrics, full_metrics = load_results()
    plot_metric_comparison(metrics)
    plot_confusion_matrices(full_metrics)
    plot_roc_curves(full_metrics)
    for model_name in PARAM_GRIDS:
        if (MODELS_DIR / f"{model_name}_pipeline.joblib").exists():
            plot_feature_importance(model_name)
    print(f"Saved figures to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all_plots()
