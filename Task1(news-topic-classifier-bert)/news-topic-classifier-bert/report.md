# Final Report: News Topic Classifier Using BERT

**Task 1 — AI/ML Engineering Advanced Internship, DevelopersHub Corporation**

## 1. Problem Statement

News aggregation platforms need to automatically route headlines into topic
sections (World, Sports, Business, Sci/Tech) at scale. This project
fine-tunes a pretrained transformer (`bert-base-uncased`) on the AG News
dataset to perform this 4-class classification, and deploys it as an
interactive tool.

## 2. Dataset

**AG News** (via HuggingFace `datasets`): 120,000 training rows and 7,600
test rows, perfectly balanced across 4 classes (30,000 train / 1,900 test
per class). Each row is a news headline concatenated with a short
description (~30–40 words total) — short enough to fit well within BERT's
128-token window with no meaningful truncation loss.

## 3. Methodology

### 3.1 Preprocessing
`AutoTokenizer.from_pretrained("bert-base-uncased")` tokenizes each example,
truncating/padding to a fixed length of 128 tokens. No manual cleaning
(lowercasing, stopword removal) is applied — BERT's WordPiece tokenizer and
pretrained embeddings already handle casing and function words better than
hand-rolled preprocessing would.

### 3.2 Model & training
`bert-base-uncased` with a sequence-classification head (4 output classes),
fine-tuned via HuggingFace `Trainer`:
- Learning rate: 2e-5, batch size 16, weight decay 0.01
- 3 epochs with early stopping (patience 2) on macro F1
- Best checkpoint (by macro F1) restored at the end of training

### 3.3 Evaluation
- **Accuracy** and **macro precision/recall/F1** on the full 7,600-row test set
- **Confusion matrix** to see which topic pairs get confused (Business vs.
  Sci/Tech is the classic AG News confusion pair, since both cover tech
  companies)
- **Per-class precision/recall/F1** bar chart
- **Confidence distribution** (correct vs. incorrect predictions) to sanity
  check that the model's confidence is well-calibrated — correct
  predictions should cluster near 1.0, incorrect ones should be more spread
  out or borderline

### 3.4 Deployment
A Streamlit app (`app/streamlit_app.py`) loads the fine-tuned model and lets
a user paste any headline/article snippet and see the predicted category
with per-class confidence scores, plus the offline evaluation dashboard
(accuracy/F1 + confusion matrix + per-class chart) once training has been
run.

## 4. How to Reproduce Results

```bash
pip install -r requirements.txt
python -m src.train              # or --quick for a fast CPU smoke test
python -m src.evaluate
python -m src.visualize
streamlit run app/streamlit_app.py
```

This project's code was developed and validated (syntax-checked, and its
metrics/plotting logic unit-tested against synthetic predictions) in an
offline sandbox without GPU/internet access to download BERT's ~440MB
weights or the AG News dataset. **Run the commands above in an environment
with internet access** (and ideally a GPU — full fine-tuning on CPU is slow
but works with `--quick` for validation) to produce the actual
`results/metrics.json`, `results/predictions.csv`, and the 3 charts
referenced below.

## 5. Expected Findings & Discussion

- **BERT-base on full AG News typically reaches ~94–95% accuracy and macro
  F1** in published benchmarks and common reproductions — this is a
  well-studied, near-saturated dataset for transformer models, so results
  in that range are expected and a good sanity check that training worked.
- **The most common confusion pair is Business vs. Sci/Tech**, since many
  articles about tech companies (earnings, stock moves, product launches)
  plausibly belong to either category — expect the confusion matrix's
  off-diagonal mass to concentrate there more than, say, World vs. Sports
  (which share almost no vocabulary).
- **Confidence calibration**: correctly classified examples should show a
  confidence distribution heavily skewed toward 1.0; misclassifications
  should show comparatively lower, more spread-out confidence — this is a
  useful check for whether the model "knows what it doesn't know," which
  matters if you'd want to route low-confidence predictions to human review
  in a production system.
- **`--quick` mode (3,000 train rows, 1 epoch)** should still reach a
  respectable ~85–90% accuracy, confirming the pipeline works correctly
  even before committing to the full multi-hour training run.

Fill in your actual numbers here once you've run the pipeline:

| Metric | Score |
|---|---|
| Accuracy | *(fill in from results/metrics.json)* |
| Precision (macro) | *(fill in)* |
| Recall (macro) | *(fill in)* |
| F1 (macro) | *(fill in)* |

## 6. Limitations

- AG News headlines are relatively clean, well-formed English text —
  real-world news feeds (social media snippets, non-English sources, OCR'd
  text) would need more robust preprocessing.
- Fixed 4-class taxonomy; extending to more granular topics (Politics,
  Health, Entertainment, etc.) would require a differently-labeled dataset.
- Only single-label classification is handled; some articles are genuinely
  multi-topic (e.g. "Sports" + "Business" for a stadium-naming-rights deal).

## 7. Conclusion

This project delivers a complete, reproducible BERT fine-tuning pipeline for
news topic classification: tokenization, training with early stopping,
thorough evaluation (accuracy/F1/confusion matrix/calibration), and a live
Streamlit deployment. The modular `src/` structure (`config.py` for all
hyperparameters) makes it straightforward to swap in a different transformer
backbone, a different dataset, or a more granular label taxonomy without
touching the training or evaluation logic.
