"""
generate_dataset.py
--------------------
Builds a labeled, free-text customer support ticket dataset.

Why synthetic-but-templated data?
Public "clean" support-ticket datasets are either tiny, gated behind a
Kaggle login, or too messy for a reproducible internship submission.
This script generates a realistic dataset by combining many phrasing
templates with randomized slot-fillers (product names, order IDs, error
codes, etc.) across 8 support categories. The result behaves like real
ticket text: variable length, informal tone, overlapping vocabulary
between classes (which is exactly what makes zero-shot vs few-shot
comparison interesting).

Run:
    python data/generate_dataset.py
Produces:
    data/support_tickets.csv   (columns: ticket_id, text, category)
"""

import random
import pandas as pd
from pathlib import Path

random.seed(42)

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

PRODUCTS = ["the mobile app", "your dashboard", "the desktop client",
            "the API", "the subscription plan", "the browser extension",
            "the checkout page", "my account settings"]

ORDER_IDS = [f"#{random.randint(100000, 999999)}" for _ in range(200)]

TEMPLATES = {
    "Billing and Payments": [
        "I was charged twice for my subscription this month, can someone check my invoice?",
        "My credit card was billed {amt} but my plan only costs {amt2}. Please refund the difference.",
        "I need an updated invoice for order {oid} for my accounting records.",
        "The payment failed but the amount was still deducted from my bank account.",
        "Can you explain why my monthly bill suddenly increased?",
        "I want to update my billing address and payment method on file.",
        "Why was I charged a late fee when I paid on time?",
        "Requesting a receipt for the payment made on {oid}.",
    ],
    "Technical Issue": [
        "{product} keeps freezing every time I try to open it.",
        "I'm getting a 500 server error whenever I load {product}.",
        "The page won't load and just shows a blank white screen.",
        "{product} crashes randomly and I lose all my unsaved work.",
        "I can't upload files, the progress bar gets stuck at 40%.",
        "Video calls keep dropping after a few minutes on {product}.",
        "The app is extremely slow today, is there an outage?",
        "I keep getting timeout errors when connecting to the server.",
    ],
    "Account Access": [
        "I forgot my password and the reset email never arrives.",
        "My account got locked after too many failed login attempts.",
        "I can't log in even though I'm entering the correct credentials.",
        "Two-factor authentication code never gets sent to my phone.",
        "Someone else seems to have access to my account, please help.",
        "I need to change the email address linked to my account.",
        "My session keeps logging me out every few minutes.",
        "How do I recover my account after losing access to my old email?",
    ],
    "Product Inquiry": [
        "Does {product} support integration with Google Sheets?",
        "What's the difference between the Pro and Enterprise plans?",
        "Is there a student discount available for your service?",
        "Can I use {product} on multiple devices at the same time?",
        "Do you offer a free trial before committing to a subscription?",
        "What languages does {product} currently support?",
        "Is there an API rate limit on the free tier?",
        "Can I export my data if I switch to a competitor?",
    ],
    "Cancellation Request": [
        "I would like to cancel my subscription effective immediately.",
        "Please cancel order {oid}, I no longer need it.",
        "How do I close my account permanently and delete my data?",
        "I want to downgrade my plan and stop future billing.",
        "Cancel my renewal, I don't want to be charged next month.",
        "I'm unsubscribing from your service, please confirm cancellation.",
        "Requesting cancellation and a refund for the unused period.",
        "Please stop auto-renewal on my account starting next cycle.",
    ],
    "Shipping and Delivery": [
        "My order {oid} was supposed to arrive yesterday but tracking shows nothing.",
        "The package arrived damaged, several items were broken inside.",
        "I received the wrong item in my order {oid}.",
        "Tracking says delivered but I never received the package.",
        "Can I change the shipping address for order {oid} before it ships?",
        "How long does international shipping usually take?",
        "My order is stuck in customs, what should I do?",
        "The courier left the package at the wrong address.",
    ],
    "Bug Report": [
        "Found a bug: the dark mode toggle resets every time I refresh {product}.",
        "There's a glitch where duplicate notifications get sent for one event.",
        "The search filter in {product} returns results from the wrong category.",
        "Clicking 'export' produces a corrupted CSV file.",
        "The calendar widget shows the wrong timezone after daylight savings.",
        "Numbers in the report are off by a factor of 10, looks like a bug.",
        "The drag-and-drop feature stopped working after the latest update.",
        "Charts fail to render when the dataset has more than 1000 rows.",
    ],
    "Feature Request": [
        "It would be great if {product} supported dark mode.",
        "Can you add bulk export functionality to {product}?",
        "Please consider adding keyboard shortcuts for common actions.",
        "It would help to have a mobile notification setting for alerts.",
        "Could you integrate {product} with Slack for team updates?",
        "Requesting the ability to schedule recurring reports automatically.",
        "A dark theme and customizable dashboards would improve usability a lot.",
        "Please add multi-language support for non-English users.",
    ],
}

CLOSERS = [
    "", " Thanks in advance.", " Please help ASAP.", " Let me know what you need from me.",
    " This is quite urgent.", " Appreciate any update on this.", " Thank you!",
]


def fill_template(t: str) -> str:
    t = t.replace("{product}", random.choice(PRODUCTS))
    t = t.replace("{oid}", random.choice(ORDER_IDS))
    t = t.replace("{amt}", f"${random.randint(20, 200)}")
    t = t.replace("{amt2}", f"${random.randint(10, 19)}")
    return t


def generate(n_per_category: int = 40) -> pd.DataFrame:
    rows = []
    ticket_id = 1
    for category, templates in TEMPLATES.items():
        for _ in range(n_per_category):
            base = random.choice(templates)
            text = fill_template(base) + random.choice(CLOSERS)
            rows.append({"ticket_id": ticket_id, "text": text, "category": category})
            ticket_id += 1
    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["ticket_id"] = range(1, len(df) + 1)
    return df


if __name__ == "__main__":
    df = generate(n_per_category=40)  # 8 categories x 40 = 320 tickets
    out_path = Path(__file__).parent / "support_tickets.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} tickets across {df['category'].nunique()} categories to {out_path}")
    print(df['category'].value_counts())
