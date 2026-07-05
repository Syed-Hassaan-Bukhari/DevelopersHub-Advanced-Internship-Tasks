"""
predict.py
-----------
Loads the exported joblib pipeline and classifies new, raw (unprocessed)
customer records. Because the exported artifact is the FULL pipeline
(preprocessing + classifier), callers never need to manually impute,
scale, or one-hot encode anything -- that's the whole point of exporting
a Pipeline instead of a bare model.
"""

from functools import lru_cache
import joblib
import pandas as pd

from src.config import MODELS_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES


@lru_cache(maxsize=1)
def load_pipeline(path=None):
    path = path or MODELS_DIR / "churn_pipeline.joblib"
    return joblib.load(path)


def predict_one(record: dict) -> dict:
    """
    record: dict with keys matching NUMERIC_FEATURES + CATEGORICAL_FEATURES,
    e.g. {"tenure": 5, "MonthlyCharges": 80.5, "TotalCharges": 400.0,
          "gender": "Female", "SeniorCitizen": "0", ...}
    Returns {"churn": "Yes"/"No", "churn_probability": float}
    """
    pipeline = load_pipeline()
    df = pd.DataFrame([record])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    proba = pipeline.predict_proba(df)[0, 1]
    pred = "Yes" if proba >= 0.5 else "No"
    return {"churn": pred, "churn_probability": float(proba)}


if __name__ == "__main__":
    sample = {
        "tenure": 3, "MonthlyCharges": 95.0, "TotalCharges": 285.0,
        "gender": "Female", "SeniorCitizen": "0", "Partner": "No", "Dependents": "No",
        "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
    print(predict_one(sample))
