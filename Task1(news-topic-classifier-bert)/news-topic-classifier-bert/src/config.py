"""
config.py
---------
Central config: paths, labels, model name, and training hyperparameters
for the AG News BERT fine-tuning task.
"""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_DIR = ROOT_DIR / "saved_model"

RESULTS_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)
MODEL_DIR.mkdir(exist_ok=True, parents=True)

# ---- Dataset -----------------------------------------------------------
# HuggingFace moved the canonical AG News repo; try the new id first and
# fall back to the legacy one so this works regardless of `datasets` version.
DATASET_CANDIDATES = ["fancyzhx/ag_news", "ag_news"]

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_LABELS = len(LABEL_NAMES)

# ---- Model & training ----------------------------------------------------
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128           # headlines + short description fit comfortably
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

# Set to a small number for a fast smoke-test run before committing to a
# full multi-hour fine-tune (AG News has 120k train / 7.6k test rows).
TRAIN_SUBSET_SIZE = None   # e.g. 5000 for a quick run; None = full dataset
EVAL_SUBSET_SIZE = None    # e.g. 1000 for a quick run; None = full test set
