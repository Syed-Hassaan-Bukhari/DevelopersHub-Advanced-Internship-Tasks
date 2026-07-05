"""
train.py
---------
Trains both candidate models (Logistic Regression, Random Forest) inside
the full preprocessing+classifier Pipeline, tunes each with GridSearchCV,
picks the better of the two by cross-validated F1, and exports the winning
fitted Pipeline as a single joblib artifact (preprocessing + model
together, so `joblib.load(...).predict(raw_dataframe)` just works on new,
unprocessed customer records).

Usage:
    python -m src.train
"""

import json
import time
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV

from src.config import (
    PARAM_GRIDS, CV_FOLDS, SCORING, MODELS_DIR, RESULTS_DIR, RANDOM_SEED,
)
from src.preprocessing import load_and_clean, get_feature_target_split, train_test_split_data
from src.pipeline import build_pipeline


def tune_model(model_name: str, X_train, y_train):
    print(f"\n{'='*60}\nTuning {model_name} with GridSearchCV "
          f"({CV_FOLDS}-fold CV, optimizing {SCORING})\n{'='*60}")
    pipeline = build_pipeline(model_name)
    grid = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRIDS[model_name],
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1,
        verbose=1,
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"Best params: {grid.best_params_}")
    print(f"Best CV {SCORING}: {grid.best_score_:.4f}  (fit time: {elapsed:.1f}s)")
    return grid, elapsed


def main():
    print("Loading and preprocessing data...")
    df = load_and_clean()
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print(f"Train: {X_train.shape} | Test: {X_test.shape} | "
          f"Train churn rate: {y_train.mean():.1%}")

    results = {}
    fitted_grids = {}

    for model_name in PARAM_GRIDS:
        grid, elapsed = tune_model(model_name, X_train, y_train)
        fitted_grids[model_name] = grid
        results[model_name] = {
            "best_params": grid.best_params_,
            "best_cv_f1": grid.best_score_,
            "fit_time_sec": elapsed,
        }

    # Pick the winner by cross-validated F1
    winner_name = max(results, key=lambda k: results[k]["best_cv_f1"])
    winner_grid = fitted_grids[winner_name]
    print(f"\n>>> Winning model: {winner_name} "
          f"(CV F1 = {results[winner_name]['best_cv_f1']:.4f})")

    # Export the full winning Pipeline (preprocessing + tuned classifier)
    model_path = MODELS_DIR / "churn_pipeline.joblib"
    joblib.dump(winner_grid.best_estimator_, model_path)
    print(f"Exported winning pipeline to {model_path}")

    # Also export the runner-up for the comparison report in evaluate.py
    for name, grid in fitted_grids.items():
        joblib.dump(grid.best_estimator_, MODELS_DIR / f"{name}_pipeline.joblib")

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(RESULTS_DIR / "training_summary.json", "w") as f:
        json.dump({"winner": winner_name, "models": results}, f, indent=2)

    # Persist the test split so evaluate.py scores on the exact same held-out data
    X_test.to_csv(RESULTS_DIR / "X_test.csv", index=False)
    y_test.to_csv(RESULTS_DIR / "y_test.csv", index=False)

    print("\nDone. Run `python -m src.evaluate` next.")


if __name__ == "__main__":
    main()
