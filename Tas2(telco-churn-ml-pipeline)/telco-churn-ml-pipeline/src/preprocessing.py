"""
preprocessing.py
-----------------
Loading/cleaning the raw Telco churn CSV, and building the reusable
scikit-learn preprocessing pipeline (imputation + scaling for numeric
features, imputation + one-hot encoding for categorical features) via
`ColumnTransformer`. This preprocessor is the first step of the full
Pipeline built in `pipeline.py`, so scaling/encoding parameters are
always fit only on training data and applied consistently at inference.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_PATH, TARGET_COL, ID_COL, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    RANDOM_SEED, TEST_SIZE,
)


def load_and_clean(path=DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and apply the light cleaning any real ingestion of
    this dataset needs (TotalCharges arrives as object/string with blanks
    for brand-new customers in the real Kaggle dataset; we replicate that
    handling here defensively even though our generator already emits
    numeric values)."""
    df = pd.read_csv(path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    df = df.drop_duplicates(subset=ID_COL).reset_index(drop=True)

    # SeniorCitizen arrives as 0/1 int in the real dataset -- treat it as
    # categorical (it's binary, not a magnitude) for the OneHotEncoder.
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

    df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})

    return df


def get_feature_target_split(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]
    return X, y


def train_test_split_data(X, y, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED):
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: median-impute + standard-scale numeric columns,
    most-frequent-impute + one-hot-encode categorical columns."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    return preprocessor


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Rows: {len(df)}  |  Churn rate: {df[TARGET_COL].mean():.1%}")
    X, y = get_feature_target_split(df)
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
