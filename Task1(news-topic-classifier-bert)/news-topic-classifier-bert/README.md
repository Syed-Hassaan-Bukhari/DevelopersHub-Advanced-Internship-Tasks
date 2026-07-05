# News Topic Classifier — Fine-tuned BERT (AG News)

**DevelopersHub Corporation — AI/ML Engineering Internship, Task 1**

Fine-tunes `bert-base-uncased` on the AG News dataset to classify news
headlines into 4 topics, evaluates with accuracy/F1, and deploys the model
through an interactive Streamlit app.

## Objective

Fine-tune a transformer model (BERT) to classify news headlines into topic
categories, evaluate with accuracy and F1-score, and deploy for live
interaction.

## Categories

`World` · `Sports` · `Business` · `Sci/Tech`

## Approach

| Step | Details |
|---|---|
| **Dataset** | AG News (HuggingFace `datasets`) — 120,000 train / 7,600 test rows, 4 balanced classes |
| **Preprocessing** | HuggingFace `AutoTokenizer` for `bert-base-uncased`, truncate/pad to 128 tokens |
| **Model** | `bert-base-uncased` + sequence classification head, fine-tuned with HF `Trainer` |
| **Training** | 3 epochs, lr=2e-5, batch size 16, early stopping on macro F1 |
| **Evaluation** | Accuracy, macro precision/recall/F1, confusion matrix, per-class metrics, confidence distribution |
| **Deployment** | Streamlit app for live single-headline classification + results dashboard |

## Project Structure

```
news-topic-classifier-bert/
├── src/
│   ├── config.py          # paths, labels, hyperparameters
│   ├── data_utils.py      # AG News loading + tokenization
│   ├── train.py           # fine-tuning entry point (HF Trainer)
│   ├── evaluate.py        # metrics + confusion matrix on test set
│   ├── predict.py         # single-text inference helper
│   └── visualize.py       # evaluation charts
├── app/
│   └── streamlit_app.py   # live demo + results dashboard
├── notebooks/
│   └── news_topic_classifier_bert.ipynb
├── saved_model/           # generated: fine-tuned model + tokenizer (git-ignored, large)
├── results/               # generated: predictions.csv, metrics.json, figures/
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

**1. Fine-tune BERT on AG News:**
```bash
python -m src.train                 # full run: 120k rows, 3 epochs (GPU strongly recommended)
python -m src.train --quick         # fast smoke test: 3k rows, 1 epoch (CPU-friendly)
```
Saves the fine-tuned model + tokenizer to `saved_model/`.

**2. Evaluate on the full test set:**
```bash
python -m src.evaluate
python -m src.visualize
```
Produces `results/metrics.json`, `results/predictions.csv`, and 3 charts in
`results/figures/`.

**3. Launch the live demo:**
```bash
streamlit run app/streamlit_app.py
```

## Key Results

Run `python -m src.evaluate` after training to reproduce; a properly
fine-tuned `bert-base-uncased` on full AG News typically reaches:

| Metric | Score |
|---|---|
| Accuracy | ~0.94–0.95 |
| Macro F1 | ~0.94–0.95 |

(These are the widely-reported reference numbers for BERT-base on AG News;
your exact run will be saved to `results/metrics.json` — see `report.md`
for a full discussion and how to fill in your actual numbers.)

## Notes

- `--quick` mode exists so you can validate the whole pipeline (train →
  evaluate → visualize → Streamlit) end-to-end on CPU in a few minutes
  before committing to the full multi-hour fine-tune.
- `saved_model/` is excluded from git via `.gitignore` (model weights are
  ~440MB) — regenerate it with `python -m src.train`, or use Git LFS if you
  want to version it.
