"""
Variant 1 dataset generator.

Topic: telecom customer churn prediction.
Output: artifacts/customer_churn.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
ROW_COUNT = 500
SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
OUTPUT_PATH = ARTIFACTS_DIR / "customer_churn.csv"


def build_customer_churn_data() -> pd.DataFrame:
    """Create a reproducible telecom churn dataset with common data issues."""
    rng = np.random.default_rng(RANDOM_SEED)

    age = rng.normal(38, 12, ROW_COUNT).clip(18, 70).round(1)
    tenure = rng.integers(1, 60, ROW_COUNT).astype(float)
    charges = np.round(rng.uniform(20, 120, ROW_COUNT), 2)
    num_services = rng.integers(0, 10, ROW_COUNT).astype(float)
    support_calls = rng.integers(0, 10, ROW_COUNT).astype(float)
    satisfaction = np.round(rng.uniform(1, 5, ROW_COUNT), 1)

    churn_prob = 1 / (1 + np.exp(tenure / 20 + satisfaction - 5))
    churn = (rng.uniform(0, 1, ROW_COUNT) < churn_prob).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": range(1001, 1001 + ROW_COUNT),
            "age": age,
            "gender": rng.choice(["Male", "Female"], ROW_COUNT),
            "region": rng.choice(["North", "South", "East", "West"], ROW_COUNT),
            "tenure_months": tenure,
            "monthly_charges": charges,
            "num_services": num_services,
            "support_calls": support_calls,
            "satisfaction_score": satisfaction,
            "contract_type": rng.choice(["Month", "One year", "Two year"], ROW_COUNT),
            "payment_method": rng.choice(["Card", "Transfer", "Cash"], ROW_COUNT),
            "churn": churn,
        }
    )

    missing_specs = {
        "age": 40,
        "support_calls": 30,
        "payment_method": 20,
        "region": 15,
        "tenure_months": 10,
        "satisfaction_score": 12,
    }
    for column, count in missing_specs.items():
        df.loc[rng.choice(ROW_COUNT, count, replace=False), column] = np.nan

    df.loc[5, "age"] = 230
    df.loc[10, "num_services"] = 99
    df.loc[15, "monthly_charges"] = -15.0
    df.loc[20, "monthly_charges"] = -22.5
    df.loc[25, "age"] = 195
    df.loc[30, "satisfaction_score"] = 12.0

    duplicates = df.iloc[:15].copy()
    return pd.concat([df, duplicates], ignore_index=True)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = build_customer_churn_data()
    df.to_csv(OUTPUT_PATH, index=False)

    print("Variant 1: telecom customer churn")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")
    print(f"Duplicate customer_id rows: {df.duplicated(subset=['customer_id']).sum()}")
    print(f"Missing values: {int(df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
