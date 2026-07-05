# Final Report: End-to-End ML Pipeline for Customer Churn Prediction

**Task 2 — AI/ML Engineering Advanced Internship, DevelopersHub Corporation**

> Unlike the LLM-based tasks in this internship (which need model downloads
> this sandbox can't reach), **this entire pipeline was actually executed**
> while building the project. Every number below is real output, not a
> placeholder — see `results/metrics.json` and `results/figures/`.

## 1. Problem Statement

Telecom providers lose significant revenue to customer churn. Identifying
at-risk customers *before* they cancel lets retention teams intervene
(discounts, contract upgrades, support outreach). This project builds a
reusable, production-ready ML pipeline that predicts churn from customer
account and service data.

## 2. Dataset

Telco Customer Churn schema (19 features: demographics, account info,
subscribed services, billing) — 4,000 customers, **30.1% churn rate**
(realistically imbalanced, similar to the real-world ~26.5% rate in the
well-known Kaggle version of this dataset). This sandbox has no internet
access to download the actual Kaggle CSV, so `data/generate_dataset.py`
generates a same-schema dataset with realistic, encoded churn-risk logic
(see README) rather than random labels — the classification problem is
genuinely non-trivial, as the results below demonstrate.

## 3. Methodology

### 3.1 Preprocessing (`ColumnTransformer` inside the `Pipeline`)
- **Numeric** (`tenure`, `MonthlyCharges`, `TotalCharges`): median imputation
  → `StandardScaler`
- **Categorical** (16 columns: demographics, services, contract, billing):
  most-frequent imputation → `OneHotEncoder(handle_unknown="ignore")`

Doing this inside a `Pipeline` (rather than transforming the DataFrame by
hand) guarantees the fitted scaler/encoder from training is reapplied
identically at inference — no train/inference skew.

### 3.2 Models & tuning
Two models, each as the final step of the same `Pipeline`:
- **Logistic Regression** — `GridSearchCV` over `C ∈ {0.01, 0.1, 1, 10}`,
  `class_weight ∈ {None, "balanced"}`
- **Random Forest** — `GridSearchCV` over `n_estimators ∈ {200, 400}`,
  `max_depth ∈ {None, 8, 16}`, `min_samples_leaf ∈ {1, 3, 5}`,
  `class_weight ∈ {None, "balanced"}`

Both tuned with 5-fold cross-validation, **optimizing F1** rather than
accuracy — with 30% churn / 70% no-churn, a model that always predicts "No
churn" would already score 70% accuracy while being completely useless for
the business goal (finding at-risk customers).

### 3.3 Model export
The pipeline with the better cross-validated F1 is exported as a single
`joblib` file (`models/churn_pipeline.joblib`) containing the full fitted
preprocessing + classifier — it accepts raw customer records directly.

## 4. Actual Results

**Cross-validated tuning (5-fold CV, training set):**

| Model | Best CV F1 | Fit time |
|---|---|---|
| Logistic Regression | **0.6404** | 1.6s |
| Random Forest | 0.6321 | 146.8s |

**Held-out test set (800 customers, never seen during tuning):**

| Metric | Logistic Regression | Random Forest |
|---|---|---|
| Accuracy | 0.7113 | 0.7275 |
| Precision | 0.5134 | 0.5387 |
| Recall | **0.7925** | 0.6639 |
| F1-score | **0.6232** | 0.5948 |
| ROC-AUC | 0.7950 | 0.7970 |

**Best hyperparameters:**
- Logistic Regression: `C=10`, `class_weight="balanced"`, `solver="lbfgs"`
- Random Forest: `n_estimators=400`, `max_depth=8`, `min_samples_leaf=1`, `class_weight="balanced"`

**Winner: Logistic Regression** (higher CV F1: 0.6404 vs 0.6321) — this is
the pipeline exported to `models/churn_pipeline.joblib`.

### Confusion matrix (Logistic Regression, test set)

|  | Predicted: No Churn | Predicted: Churn |
|---|---|---|
| **Actual: No Churn** | 378 | 181 |
| **Actual: Churn** | 50 | 191 |

Only 50 of 241 actual churners were missed (recall 79.3%) — at the cost of
181 false alarms (customers flagged as at-risk who wouldn't have churned).
That's the right tradeoff for churn prediction: a false alarm costs a
retention discount offered to a customer who'd have stayed anyway; a missed
churner costs the customer entirely.

### Top predictive features (see `results/figures/feature_importance_logistic_regression.png`)
`Contract_Month-to-month`, `InternetService_Fiber optic`, and
`Contract_Two year` (negatively) dominate — matching well-established churn
research: month-to-month customers face zero switching cost, while
long-term contracts and bundled fiber service create lock-in.

## 5. Why Logistic Regression beat Random Forest here

With only ~4,000 rows and mostly categorical, fairly linear-separable churn
signal (contract type and internet service dominate almost additively),
Logistic Regression's simplicity is an advantage rather than a limitation —
it doesn't have enough data to benefit from Random Forest's extra
flexibility, and `class_weight="balanced"` handles the class imbalance
directly. Random Forest's edge in accuracy (0.7275 vs 0.7113) comes from
being more conservative on churn predictions (higher precision, lower
recall) — worse for this business use case, where catching more true
churners (even at some false-alarm cost) is the goal.

## 6. How to Reproduce

```bash
pip install -r requirements.txt
python -m src.train        # ~2.5 minutes total (dominated by Random Forest grid search)
python -m src.evaluate
python -m src.visualize
streamlit run app/streamlit_app.py
```

## 7. Limitations

- Synthetic dataset (see README) — validated against a real Kaggle-style
  Telco churn dataset if available, results should generalize directionally
  (contract type and internet service type are consistently top churn
  predictors in real published analyses of this dataset) but exact
  numbers will differ.
- Only 2 models compared; gradient boosting (XGBoost/LightGBM) commonly
  outperforms both on tabular churn data and would be a natural next step.
- No feature engineering beyond the raw schema (e.g. no
  tenure-times-charges interaction terms, no service-count aggregate) —
  the `ColumnTransformer` step is the natural place to add these.

## 8. Conclusion

This project delivers a complete, genuinely-executed ML pipeline: reusable
`Pipeline`-based preprocessing, two hyperparameter-tuned models compared on
a business-appropriate metric (F1, not accuracy), a single deployable
`joblib` artifact, and a Streamlit interface for live churn-risk scoring.
The `src/config.py` module centralizes every path, feature list, and
hyperparameter grid, so swapping in the real Kaggle dataset or adding a
third model requires touching only one file.
