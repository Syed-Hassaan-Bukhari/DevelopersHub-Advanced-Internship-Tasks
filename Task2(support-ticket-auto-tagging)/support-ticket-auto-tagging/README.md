# Support Ticket Auto-Tagging with LLMs (Zero-shot vs Few-shot)

**DevelopersHub Corporation — AI/ML Engineering Internship, Task 5**

Automatically classify free-text customer support tickets into one of 8
categories using two LLM-based techniques — **zero-shot classification**
(NLI-based, no examples needed) and **few-shot classification**
(in-context learning with a handful of labeled examples) — and compare
their performance head to head.

## Objective

Build an LLM-powered auto-tagging system that:
- Classifies support tickets without any task-specific fine-tuning
- Compares zero-shot vs few-shot prompting strategies
- Returns the **top-3 most probable categories** per ticket, ranked by confidence
- Is evaluated with accuracy, precision, recall, and F1-score
- Ships with a Streamlit demo for live interaction

## Categories

`Billing and Payments` · `Technical Issue` · `Account Access` ·
`Product Inquiry` · `Cancellation Request` · `Shipping and Delivery` ·
`Bug Report` · `Feature Request`

## Approach

| Component | Technique |
|---|---|
| **Zero-shot** | `facebook/bart-large-mnli` via HuggingFace's `zero-shot-classification` pipeline. Each category is turned into a hypothesis ("This support ticket is about *X*.") and scored by NLI entailment — no examples required. |
| **Few-shot** | `google/flan-t5-base`. A prompt is built with 1–2 labeled example tickets per category, then each candidate category's log-likelihood as the continuation is scored directly (length-normalized), softmaxed into a ranked probability distribution. |
| **Ranking** | Both methods output a full probability distribution over all 8 categories — we take the top-3. |
| **Evaluation** | Accuracy, macro precision/recall/F1, confusion matrices, and a top-3 "hit rate" (was the true label anywhere in the top 3?). |

Why this few-shot design instead of free-form generation? Asking the LLM
to *generate* a category name is hard to parse reliably and gives no
confidence score. Scoring each candidate label's likelihood as a forced
continuation (the same technique used in the GPT-3 paper's few-shot
evaluations) gives a clean, comparable ranking — required for the
top-3 output.

## Project Structure

```
support-ticket-auto-tagging/
├── data/
│   ├── generate_dataset.py     # synthetic ticket dataset generator
│   └── support_tickets.csv     # 8 categories x 40 tickets
├── src/
│   ├── config.py                # paths, labels, model names, few-shot examples
│   ├── preprocessing.py         # loading, cleaning, train/test split
│   ├── zero_shot_classifier.py  # NLI-based zero-shot classifier
│   ├── few_shot_classifier.py   # LLM log-likelihood few-shot classifier
│   ├── evaluate.py               # metrics computation, saves results/
│   ├── visualize.py              # comparison charts, confusion matrices
│   └── main.py                   # end-to-end pipeline entry point
├── app/
│   └── streamlit_app.py          # interactive demo + results dashboard
├── notebooks/
│   └── support_ticket_tagging.ipynb   # exploratory / narrative walkthrough
├── results/                      # generated: predictions.csv, metrics.json, figures/
├── requirements.txt
├── README.md
└── report.md                     # final report & conclusions
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Generate the dataset** (already included as `data/support_tickets.csv`, regenerate if needed):
```bash
python data/generate_dataset.py
```

**2. Run the full evaluation pipeline** (classifies the test split with both
methods, computes metrics, generates all charts):
```bash
python -m src.main                # full test split (~78 tickets)
python -m src.main --n 30         # quick run on 30 sampled tickets
python -m src.main --shots 2      # 2 few-shot examples per category
```
Outputs land in `results/`: `predictions.csv`, `metrics.json`, `figures/*.png`.

**3. Launch the interactive demo:**
```bash
streamlit run app/streamlit_app.py
```

## Key Results

Run `python -m src.main` to reproduce; a representative run produces:

| Metric | Zero-shot | Few-shot |
|---|---|---|
| Accuracy | ~0.70–0.80 | ~0.60–0.75 |
| Macro F1 | ~0.68–0.78 | ~0.58–0.73 |
| Top-3 Accuracy | ~0.90+ | ~0.85+ |

(Exact numbers depend on model versions and hardware; see `results/metrics.json`
after running the pipeline, and `report.md` for a full discussion.)

**Headline takeaway:** zero-shot NLI classification is a strong, dependency-free
baseline for well-separated, clearly-named categories like these support-ticket
tags, since the category names themselves carry most of the semantic signal an
NLI model needs. Few-shot prompting mainly helps when category boundaries are
ambiguous or non-obvious from the label name alone — see `report.md`.

## Notes on the dataset

Public support-ticket datasets are either tiny, gated, or too noisy for a
reproducible submission, so `data/generate_dataset.py` builds a labeled dataset
from realistic phrasing templates across the 8 categories (320 tickets, later
deduplicated). Swap in a real dataset by pointing `src/config.DATA_PATH` at any
CSV with `text` and `category` columns matching the categories in `CATEGORIES`.
