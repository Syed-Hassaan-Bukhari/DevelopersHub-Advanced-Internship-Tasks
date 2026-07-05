"""
main.py
--------
End-to-end pipeline entry point:
  1. Load & preprocess the support ticket dataset
  2. Run zero-shot classification on the test split
  3. Run few-shot classification on the test split
  4. Evaluate both (accuracy, precision, recall, F1, top-3 accuracy)
  5. Generate all comparison visualizations
  6. Print a final summary

Usage:
    python -m src.main                     # full test split
    python -m src.main --n 40              # quick run on 40 sampled tickets
    python -m src.main --shots 2           # 2 few-shot examples per category
"""

import argparse
import json

from src.evaluate import run_full_evaluation
from src.visualize import generate_all_plots
from src.config import RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description="Support ticket auto-tagging pipeline")
    parser.add_argument("--n", type=int, default=None,
                         help="Optional: limit evaluation to N sampled test tickets (faster runs)")
    parser.add_argument("--shots", type=int, default=1,
                         help="Few-shot examples per category (default: 1)")
    args = parser.parse_args()

    print("=" * 70)
    print("SUPPORT TICKET AUTO-TAGGING: ZERO-SHOT vs FEW-SHOT LLM CLASSIFICATION")
    print("=" * 70)

    zs_metrics, fs_metrics, predictions_df = run_full_evaluation(
        n_test_samples=args.n, shots_per_category=args.shots
    )

    print("\nGenerating visualizations...")
    generate_all_plots()

    winner = "Few-shot" if fs_metrics["f1_macro"] > zs_metrics["f1_macro"] else "Zero-shot"
    print("\n" + "=" * 70)
    print(f"SUMMARY: {winner} achieved the higher macro F1-score on this run.")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
