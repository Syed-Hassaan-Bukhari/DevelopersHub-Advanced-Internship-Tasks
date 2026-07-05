"""
train.py
---------
Fine-tunes bert-base-uncased on AG News for 4-class topic classification
using the HuggingFace Trainer API.

Usage:
    python -m src.train                  # full dataset, 3 epochs
    python -m src.train --quick          # small subset, 1 epoch (smoke test)
    python -m src.train --epochs 2 --batch_size 32
"""

import argparse
import numpy as np
import evaluate as hf_evaluate
from transformers import (
    AutoModelForSequenceClassification, TrainingArguments, Trainer,
    EarlyStoppingCallback,
)

from src.config import (
    MODEL_NAME, NUM_LABELS, LABEL_NAMES, LEARNING_RATE, BATCH_SIZE,
    NUM_EPOCHS, WEIGHT_DECAY, MODEL_DIR, RANDOM_SEED,
)
from src.data_utils import get_train_eval_splits, get_tokenizer

accuracy_metric = hf_evaluate.load("accuracy")
f1_metric = hf_evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
    f1_weighted = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT on AG News")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--quick", action="store_true",
                         help="Fast smoke test: 3000 train / 500 eval rows, 1 epoch")
    args = parser.parse_args()

    if args.quick:
        import src.config as cfg
        cfg.TRAIN_SUBSET_SIZE = 3000
        cfg.EVAL_SUBSET_SIZE = 500
        args.epochs = 1

    print("Loading and tokenizing AG News...")
    raw, tokenized = get_train_eval_splits()
    tokenizer = get_tokenizer()

    print(f"Train rows: {len(tokenized['train'])} | Test rows: {len(tokenized['test'])}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS,
        id2label={i: l for i, l in enumerate(LABEL_NAMES)},
        label2id={l: i for i, l in enumerate(LABEL_NAMES)},
    )

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=WEIGHT_DECAY,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=50,
        seed=RANDOM_SEED,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("Starting fine-tuning...")
    trainer.train()

    print("\nFinal evaluation on test set:")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"\nSaving model + tokenizer to {MODEL_DIR}")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    import json
    with open(MODEL_DIR / "final_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
