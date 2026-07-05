"""
streamlit_app.py
------------------
Interactive demo for the Telco churn prediction pipeline.

Run with:
    streamlit run app/streamlit_app.py

Requires the exported pipeline at models/churn_pipeline.joblib
(produced by `python -m src.train`).
"""

import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import MODELS_DIR, RESULTS_DIR

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉", layout="wide")

st.title("📉 Telco Customer Churn Predictor")
st.caption("End-to-end scikit-learn Pipeline: preprocessing + tuned classifier")

if not (MODELS_DIR / "churn_pipeline.joblib").exists():
    st.error("No trained pipeline found. Run `python -m src.train` first.")
    st.stop()

from src.predict import predict_one

with st.sidebar:
    st.header("Customer profile")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0)
    total_charges = st.number_input("Total charges ($)", 0.0, 10000.0, float(tenure * monthly_charges))

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior citizen", ["0", "1"])
    partner = st.selectbox("Has partner", ["Yes", "No"])
    dependents = st.selectbox("Has dependents", ["Yes", "No"])

    phone_service = st.selectbox("Phone service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming movies", ["Yes", "No", "No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

record = {
    "tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
    "PhoneService": phone_service, "MultipleLines": multiple_lines,
    "InternetService": internet_service, "OnlineSecurity": online_security,
    "OnlineBackup": online_backup, "DeviceProtection": device_protection,
    "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
    "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment_method,
}

st.subheader("Prediction")
if st.button("Predict churn risk", type="primary"):
    result = predict_one(record)
    prob = result["churn_probability"]

    col1, col2 = st.columns(2)
    col1.metric("Predicted", result["churn"])
    col2.metric("Churn probability", f"{prob:.1%}")

    if prob >= 0.5:
        st.warning("⚠️ High churn risk — consider retention outreach (discount, contract upgrade offer).")
    else:
        st.success("✅ Low churn risk.")

    st.progress(min(prob, 1.0))

st.markdown("---")
st.subheader("Model evaluation results")

metrics_path = RESULTS_DIR / "metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

    for model_name, m in metrics.items():
        st.markdown(f"**{model_name.replace('_', ' ').title()}**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['accuracy']:.2%}")
        c2.metric("Precision", f"{m['precision']:.2%}")
        c3.metric("Recall", f"{m['recall']:.2%}")
        c4.metric("F1-score", f"{m['f1']:.2%}")
        c5.metric("ROC-AUC", f"{m['roc_auc']:.2%}")

    fig_path = RESULTS_DIR / "figures" / "metric_comparison.png"
    if fig_path.exists():
        st.image(str(fig_path))
else:
    st.info("Run `python -m src.evaluate` and `python -m src.visualize` to populate this section.")
