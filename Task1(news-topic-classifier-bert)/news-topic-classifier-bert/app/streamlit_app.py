"""
streamlit_app.py
------------------
Interactive demo for the fine-tuned BERT news topic classifier.

Run with:
    streamlit run app/streamlit_app.py

Requires a fine-tuned model in saved_model/ (produced by `python -m src.train`).
"""

import sys
import json
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import LABEL_NAMES, MODEL_DIR, RESULTS_DIR

st.set_page_config(page_title="News Topic Classifier (BERT)", page_icon="📰", layout="wide")

st.title("📰 News Topic Classifier — Fine-tuned BERT")
st.caption("bert-base-uncased fine-tuned on AG News (World / Sports / Business / Sci-Tech)")

if not (MODEL_DIR / "config.json").exists():
    st.error(
        "No fine-tuned model found in `saved_model/`. Run `python -m src.train` "
        "(or `python -m src.train --quick` for a fast smoke test) first."
    )
    st.stop()

from src.predict import predict  # imported after the model-existence check

st.subheader("Classify a headline or short article")
default_text = "Apple unveiled its latest chip, promising major gains in on-device AI performance."
text = st.text_area("Enter news text:", value=default_text, height=120)

if st.button("Classify", type="primary") and text.strip():
    with st.spinner("Running BERT inference..."):
        ranked = predict(text)

    top = ranked[0]
    st.success(f"**Predicted category: {top['label']}**  (confidence: {top['score']:.1%})")

    import pandas as pd
    df = pd.DataFrame(ranked)
    df["score"] = df["score"].round(4)
    col1, col2 = st.columns([1, 1.4])
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col2:
        st.bar_chart(df.set_index("label")["score"])

st.markdown("---")
st.subheader("Model evaluation results")

metrics_path = RESULTS_DIR / "metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    col2.metric("Precision (macro)", f"{metrics['precision_macro']:.2%}")
    col3.metric("F1-score (macro)", f"{metrics['f1_macro']:.2%}")

    cm_path = RESULTS_DIR / "figures" / "confusion_matrix.png"
    pc_path = RESULTS_DIR / "figures" / "per_class_metrics.png"
    c1, c2 = st.columns(2)
    if cm_path.exists():
        c1.image(str(cm_path), caption="Confusion matrix")
    if pc_path.exists():
        c2.image(str(pc_path), caption="Per-class precision/recall/F1")
else:
    st.info("Run `python -m src.evaluate` and `python -m src.visualize` to populate this section.")
