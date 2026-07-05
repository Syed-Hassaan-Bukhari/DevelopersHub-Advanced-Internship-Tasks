"""
config.py
---------
Single source of truth for paths, category labels, model names, and
few-shot example prompts used across the whole project.
"""

from pathlib import Path

# ---- Paths -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "support_tickets.csv"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# ---- Labels ------------------------------------------------------------
CATEGORIES = [
    "Billing and Payments",
    "Technical Issue",
    "Account Access",
    "Product Inquiry",
    "Cancellation Request",
    "Shipping and Delivery",
    "Bug Report",
    "Feature Request",
]

# ---- Models --------------------------------------------------------------
# Zero-shot: an NLI model used as an off-the-shelf zero-shot classifier.
ZERO_SHOT_MODEL = "facebook/bart-large-mnli"

# Few-shot: a small instruction-tuned seq2seq LLM used with in-context
# examples. Small enough to run on CPU for an internship-scale project.
FEW_SHOT_MODEL = "google/flan-t5-base"

# Hypothesis template used by the zero-shot NLI pipeline.
ZERO_SHOT_HYPOTHESIS_TEMPLATE = "This support ticket is about {}."

# ---- Few-shot example bank -------------------------------------------
# 2 hand-picked example tickets per category, used to build the few-shot
# prompt. These are deliberately NOT drawn from the evaluation split
# (see src/preprocessing.py) to avoid leakage.
FEW_SHOT_EXAMPLES = {
    "Billing and Payments": [
        "I was charged twice for the same order, can you refund one of the charges?",
        "My invoice shows an amount that doesn't match my subscription plan.",
    ],
    "Technical Issue": [
        "The app crashes every time I open the reports section.",
        "I keep getting a connection timeout error when loading the dashboard.",
    ],
    "Account Access": [
        "I can't log into my account even with the correct password.",
        "The password reset link in my email doesn't work.",
    ],
    "Product Inquiry": [
        "Does your platform integrate with Salesforce?",
        "What is included in the Enterprise pricing tier?",
    ],
    "Cancellation Request": [
        "Please cancel my subscription starting next billing cycle.",
        "I want to close my account and stop all future charges.",
    ],
    "Shipping and Delivery": [
        "My package tracking hasn't updated in five days.",
        "I received someone else's order instead of mine.",
    ],
    "Bug Report": [
        "The export button generates a file with no data in it.",
        "Dark mode doesn't apply to the settings page, that looks like a bug.",
    ],
    "Feature Request": [
        "It would be great to have a keyboard shortcut for saving drafts.",
        "Can you add an option to schedule recurring exports?",
    ],
}

RANDOM_SEED = 42
TEST_SIZE = 0.3  # fraction of tickets held out for evaluation
