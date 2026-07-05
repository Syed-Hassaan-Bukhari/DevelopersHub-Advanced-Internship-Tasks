# End-to-End ML Pipeline — Telco Customer Churn Prediction

**DevelopersHub Corporation — AI/ML Engineering Internship, Task 2**

A production-style, reusable scikit-learn `Pipeline` that predicts customer
churn: preprocessing (imputation, scaling, one-hot encoding), two tuned
models (Logistic Regression and Random Forest via `GridSearchCV`), full
evaluation, and a single exported `joblib` artifact ready for deployment.

**This entire pipeline was actually run end-to-end while building this
project** (not just written) — the numbers in this README and `report.md`
are real output from `results/metrics.json`, not placeholders.

## Objective

Build a reusable, production-ready ML pipeline for predicting customer
churn using scikit-learn's `Pipeline` API, tune hyperparameters with
`GridSearchCV`, and export the complete pipeline with `joblib`.

## Approach

| Step | Details |
|---|---|
| **Dataset** | Telco Customer Churn schema (19 features + target), 4,000 customers, 30.1% churn rate |
| **Preprocessing** | `ColumnTransformer`: median-impute + `StandardScaler` for numeric columns; most-frequent-impute + `OneHotEncoder` for categorical columns — all inside the `Pipeline` |
| **Models** | Logistic Regression and Random Forest, each wrapped in the same `Pipeline` (preprocessing + classifier) |
| **Tuning** | `GridSearchCV`, 5-fold CV, optimizing F1 (chosen over accuracy given class imbalance) |
| **Export** | Winning pipeline saved as a single `joblib` file — takes raw, unprocessed customer records straight in |
| **Evaluation** | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, ROC curve, feature importance |

## Project Structure

```
telco-churn-ml-pipeline/
├── data/
│   ├── generate_dataset.py     # synthetic Telco-schema dataset generator
│   └── telco_churn.csv         # 4,000 customers
├── src/
│   ├── config.py                # paths, feature lists, hyperparameter grids
│   ├── preprocessing.py         # loading, cleaning, ColumnTransformer
│   ├── pipeline.py              # Pipeline construction (preprocessing + classifier)
│   ├── train.py                 # GridSearchCV tuning + joblib export
│   ├── evaluate.py              # metrics on held-out test set
│   ├── visualize.py             # comparison charts
│   └── predict.py               # single-customer inference helper
├── app/
│   └── streamlit_app.py         # interactive churn-risk calculator
├── notebooks/
│   └── telco_churn_pipeline.ipynb
├── models/                      # generated: churn_pipeline.joblib (+ per-model variants)
├── results/                     # generated: metrics.json, figures/
├── requirements.txt
├── README.md
└── report.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python data/generate_dataset.py   # regenerate the dataset (already included)
python -m src.train                # GridSearchCV both models, export winning pipeline
python -m src.evaluate              # metrics on held-out test set
python -m src.visualize              # comparison charts
streamlit run app/streamlit_app.py    # interactive demo
```

**Using the exported pipeline elsewhere** — this is the whole point of
exporting a full `Pipeline` rather than a bare model:
```python
import joblib
import pandas as pd

pipeline = joblib.load("models/churn_pipeline.joblib")
new_customer = pd.DataFrame([{
    "tenure": 3, "MonthlyCharges": 95.0, "TotalCharges": 285.0,
    "gender": "Female", "SeniorCitizen": "0", "Partner": "No", "Dependents": "No",
    "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}])
pipeline.predict_proba(new_customer)  # no manual scaling/encoding needed
```

## Actual Results (from this run — see `results/metrics.json`)

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.711 | 0.728 |
| Precision | 0.513 | 0.539 |
| Recall | 0.793 | 0.664 |
| F1-score | **0.623** (winner, by CV F1) | 0.595 |
| ROC-AUC | 0.795 | 0.797 |

**Best hyperparameters found:**
- Logistic Regression: `C=10`, `class_weight=balanced`, `solver=lbfgs`
- Random Forest: `n_estimators=400`, `max_depth=8`, `min_samples_leaf=1`, `class_weight=balanced`

**Logistic Regression was selected** as the exported pipeline (higher
cross-validated F1: 0.640 vs. 0.632). See `report.md` for full discussion,
including why recall matters more than raw accuracy for churn prediction.

## Notes on the dataset

This sandbox has no internet access to download the real Kaggle "Telco
Customer Churn" dataset, so `data/generate_dataset.py` generates a
same-schema synthetic dataset with realistic, encoded risk relationships
(month-to-month contracts, fiber optic without tech support, and
electronic-check payment all increase churn probability; long tenure and
long-term contracts decrease it) — a genuinely non-trivial classification
problem, not toy data. Swap in the real dataset by pointing
`src/config.DATA_PATH` at any CSV matching the same column schema.
