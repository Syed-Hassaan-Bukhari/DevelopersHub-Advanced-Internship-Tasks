"""
generate_dataset.py
--------------------
Builds a realistic Telco Customer Churn dataset matching the schema of the
widely-used Kaggle "Telco Customer Churn" dataset (customerID, demographics,
account info, services subscribed, charges, and the Churn target).

Why generate instead of downloading? This sandbox has no internet access,
and the goal is a fully reproducible submission that runs end-to-end
without depending on an external download link that may change or require
a Kaggle login. The generation logic encodes realistic relationships found
in the real dataset (e.g. month-to-month contracts + fiber optic + no tech
support => much higher churn probability; long tenure + long-term contract
=> much lower churn probability) so the ML pipeline faces a genuinely
non-trivial, realistic classification problem.

Run:
    python data/generate_dataset.py
Produces:
    data/telco_churn.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 4000


def generate(n=N) -> pd.DataFrame:
    gender = RNG.choice(["Male", "Female"], size=n)
    senior_citizen = RNG.choice([0, 1], size=n, p=[0.84, 0.16])
    partner = RNG.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = RNG.choice(["Yes", "No"], size=n, p=[0.3, 0.7])

    tenure = RNG.integers(0, 73, size=n)

    phone_service = RNG.choice(["Yes", "No"], size=n, p=[0.9, 0.1])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        RNG.choice(["Yes", "No"], size=n, p=[0.42, 0.58])
    )

    internet_service = RNG.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22])

    def dependent_internet_feature(p_yes=0.4):
        return np.where(
            internet_service == "No", "No internet service",
            RNG.choice(["Yes", "No"], size=n, p=[p_yes, 1 - p_yes])
        )

    online_security = dependent_internet_feature(0.29)
    online_backup = dependent_internet_feature(0.35)
    device_protection = dependent_internet_feature(0.35)
    tech_support = dependent_internet_feature(0.29)
    streaming_tv = dependent_internet_feature(0.39)
    streaming_movies = dependent_internet_feature(0.39)

    contract = RNG.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.21, 0.24])
    paperless_billing = RNG.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment_method = RNG.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n, p=[0.34, 0.23, 0.22, 0.21]
    )

    # Base monthly charge depends on internet service + add-ons
    base = np.select(
        [internet_service == "No", internet_service == "DSL", internet_service == "Fiber optic"],
        [20.0, 55.0, 80.0],
    )
    addon_cost = (
        (phone_service == "Yes") * RNG.uniform(2, 6, n)
        + (online_security == "Yes") * RNG.uniform(3, 7, n)
        + (online_backup == "Yes") * RNG.uniform(3, 7, n)
        + (device_protection == "Yes") * RNG.uniform(3, 7, n)
        + (tech_support == "Yes") * RNG.uniform(3, 7, n)
        + (streaming_tv == "Yes") * RNG.uniform(5, 10, n)
        + (streaming_movies == "Yes") * RNG.uniform(5, 10, n)
    )
    monthly_charges = np.round(base + addon_cost + RNG.normal(0, 3, n), 2)
    monthly_charges = np.clip(monthly_charges, 18.0, 120.0)

    total_charges = np.round(monthly_charges * tenure + RNG.normal(0, 20, n), 2)
    total_charges = np.clip(total_charges, 0, None)
    # New customers (tenure 0) commonly have blank/zero total charges in the real dataset
    total_charges = np.where(tenure == 0, 0.0, total_charges)

    # ---- Churn probability model (encodes realistic risk factors) --------
    logit = (
        -1.6
        + 1.4 * (contract == "Month-to-month")
        + 0.3 * (contract == "One year")
        + 0.9 * (internet_service == "Fiber optic")
        - 0.5 * (internet_service == "No")
        + 0.6 * (tech_support == "No") * (internet_service != "No")
        + 0.5 * (online_security == "No") * (internet_service != "No")
        + 0.4 * (payment_method == "Electronic check")
        + 0.35 * (paperless_billing == "Yes")
        + 0.25 * (senior_citizen == 1)
        - 0.35 * (partner == "Yes")
        - 0.25 * (dependents == "Yes")
        - 0.045 * tenure
        + 0.01 * (monthly_charges - 65)
    )
    prob_churn = 1 / (1 + np.exp(-logit))
    churn = RNG.binomial(1, prob_churn)
    churn_label = np.where(churn == 1, "Yes", "No")

    customer_id = [f"{RNG.integers(1000,9999)}-{''.join(RNG.choice(list('ABCDEFGHJKLMNPQRSTUVWXYZ'), 5))}"
                   for _ in range(n)]

    df = pd.DataFrame({
        "customerID": customer_id,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn_label,
    })
    return df


if __name__ == "__main__":
    df = generate()
    out_path = "data/telco_churn.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Churn rate: {(df['Churn'] == 'Yes').mean():.1%}")
    print(df.head())
