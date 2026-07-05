"""
data_utils.py
--------------
Loads the AG News dataset, tokenizes it for BERT, and exposes a few
helpers used by both training and the notebook/Streamlit demo.

AG News has 4 balanced topic classes: World, Sports, Business, Sci/Tech.
Each row is a news headline + short description (~30-40 words) -- a good
fit for BERT's [CLS]-token sequence classification head.
"""

from functools import lru_cache

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

from src.config import (
    DATASET_CANDIDATES, MODEL_NAME, MAX_LENGTH,
    TRAIN_SUBSET_SIZE, EVAL_SUBSET_SIZE, RANDOM_SEED, LABEL_NAMES,
)


def load_ag_news() -> DatasetDict:
    """Load AG News, trying the current HF Hub repo id first, then the
    legacy id (older `datasets` cache / versions still expect it)."""
    last_err = None
    for repo_id in DATASET_CANDIDATES:
        try:
            return load_dataset(repo_id)
        except Exception as e:  # noqa: BLE001 - we want to try the next candidate
            last_err = e
            continue
    raise RuntimeError(
        f"Could not load AG News from any of {DATASET_CANDIDATES}. "
        f"Last error: {last_err}"
    )


@lru_cache(maxsize=1)
def get_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataset(dataset: DatasetDict) -> DatasetDict:
    """Tokenizes the `text` column into input_ids/attention_mask, keeps
    the `label` column, and drops the raw text (BERT only needs token ids)."""
    tokenizer = get_tokenizer()

    def _tokenize(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH
        )

    tokenized = dataset.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized


def get_train_eval_splits():
    """Loads, optionally subsamples (for fast smoke tests), and tokenizes
    the AG News train/test splits. Returns (raw_dataset, tokenized_dataset)."""
    raw = load_ag_news()

    if TRAIN_SUBSET_SIZE:
        raw["train"] = raw["train"].shuffle(seed=RANDOM_SEED).select(range(TRAIN_SUBSET_SIZE))
    if EVAL_SUBSET_SIZE:
        raw["test"] = raw["test"].shuffle(seed=RANDOM_SEED).select(range(EVAL_SUBSET_SIZE))

    tokenized = tokenize_dataset(raw)
    return raw, tokenized


def id_to_label(label_id: int) -> str:
    return LABEL_NAMES[label_id]


if __name__ == "__main__":
    raw, tokenized = get_train_eval_splits()
    print(f"Train rows: {len(raw['train'])}  |  Test rows: {len(raw['test'])}")
    print("Sample:", raw["train"][0])
