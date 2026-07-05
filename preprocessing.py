"""
preprocessing.py
-----------------
Loading, cleaning, and splitting the support ticket dataset.

LLM-based classifiers (zero-shot NLI, few-shot prompting) work directly
on raw text, so preprocessing here is intentionally light: we don't
lowercase, remove stopwords, or stem, because doing so would strip
away information the LLM actually uses (casing signals like "ASAP",
punctuation, named entities). We only do the cleaning that any real
ticket-ingestion pipeline would need: whitespace normalization, control
character removal, and de-duplication.
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, RANDOM_SEED, TEST_SIZE, CATEGORIES


def clean_text(text: str) -> str:
    """Light, LLM-safe text cleaning."""
    text = str(text)
    text = re.sub(r"\s+", " ", text)          # collapse whitespace
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)  # control chars
    return text.strip()


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV, clean text, drop empties/duplicates."""
    df = pd.read_csv(path)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    unknown = set(df["category"]) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Found labels not in CATEGORIES: {unknown}")

    return df


def split_dataset(df: pd.DataFrame, test_size: float = TEST_SIZE,
                   seed: int = RANDOM_SEED):
    """Stratified train/test split (train split unused by zero-shot,
    used only as the pool few-shot examples were hand-picked to avoid)."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["category"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_dataset()
    train_df, test_df = split_dataset(df)
    print(f"Total tickets: {len(df)}")
    print(f"Train: {len(train_df)}  |  Test: {len(test_df)}")
    print(test_df["category"].value_counts())
