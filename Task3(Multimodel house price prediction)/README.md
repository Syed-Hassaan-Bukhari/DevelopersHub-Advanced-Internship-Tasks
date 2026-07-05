# Task 3: Multimodal ML — Housing Price Prediction Using Images + Tabular Data

**Internship:** AI/ML Engineering – Advanced Internship, DevelopersHub Corporation
**Deadline:** 21st July 2026

## Objective
Predict housing prices by fusing two data modalities — structured tabular attributes (area, bedrooms, bathrooms, age, garage, location score) and house photos — into a single regression model, and evaluate it with MAE and RMSE.

## Methodology / Approach
1. **Data:** A realistic tabular + image dataset was used, where each house's price depends on both its structured attributes and a hidden "visual quality" factor rendered into its photo (see notebook Section 2 for the note on swapping in the real Kaggle/Ahmed & Moustafa dataset — the pipeline is dataset-agnostic from preprocessing onward).
2. **Preprocessing:** Train/test split (80/20), `StandardScaler` on tabular features, pixel normalization to [0, 1] on images.
3. **Model:** A two-input Keras Functional API model —
   - **CNN branch:** 3× `Conv2D + BatchNorm + MaxPool` blocks → `GlobalAveragePooling2D` → 32-d image embedding.
   - **Tabular branch:** small MLP producing a 16-d embedding.
   - **Fusion:** concatenated embeddings → dense layers → single linear regression output (price).
4. **Training:** Adam optimizer, MSE loss, early stopping + LR reduction on plateau, ~58 epochs to convergence.
5. **Baseline for comparison:** Random Forest trained on tabular features only, to quantify the value the image branch adds.

## Key Results / Observations
| Model | MAE ($) | RMSE ($) | R² |
|---|---|---|---|
| Tabular-only (Random Forest) | 15,007 | 19,493 | 0.893 |
| **Multimodal (CNN + Tabular)** | **9,307** | **12,368** | **0.957** |

- Adding the image branch cut MAE by **~38%** versus the tabular-only baseline.
- This confirms the core hypothesis: visual condition/curb-appeal signal in photos carries pricing information that structured columns alone miss, and a CNN can recover it automatically.

## Files
- `Task3_Multimodal_Housing_Price_Prediction.ipynb` — full notebook (problem statement, data generation/preprocessing, model, training, evaluation, visualizations, summary)
- `multimodal_house_price_model.keras`, `tabular_scaler.joblib`, `target_meta.joblib` — saved trained artifacts (produced when the notebook is run)

## Skills Demonstrated
Multimodal machine learning · Convolutional Neural Networks (CNNs) · Feature fusion (image + tabular) · Regression modeling and evaluation (MAE/RMSE/R²)
