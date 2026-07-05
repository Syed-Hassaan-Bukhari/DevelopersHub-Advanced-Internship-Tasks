"""
pipeline.py
------------
Builds the full, exportable scikit-learn Pipeline: preprocessing
(ColumnTransformer) -> classifier. Using Pipeline end-to-end (rather than
transforming data separately) guarantees the exact same imputation/scaling/
encoding is applied at training and inference time, and lets the whole
thing be exported as a single joblib artifact with `export_pipeline`.
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.preprocessing import build_preprocessor
from src.config import RANDOM_SEED


def build_pipeline(model_name: str) -> Pipeline:
    """model_name: 'logistic_regression' or 'random_forest'."""
    preprocessor = build_preprocessor()

    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    return pipeline
