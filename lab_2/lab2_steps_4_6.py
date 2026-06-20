"""
Laboratory work 2. Outliers, categorical encoding, and feature engineering.

Variant 1: telecom customer churn.
Input: ../lab_1/artifacts/data_after_step3.csv
Output: artifacts/data_after_step6.csv and diagnostic plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_string_dtype


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
INPUT_PATH = ROOT_DIR / "lab_1" / "artifacts" / "data_after_step3.csv"
OUTPUT_PATH = ARTIFACTS_DIR / "data_after_step6.csv"
ID_COLUMN = "customer_id"
TARGET_COLUMN = "churn"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150


def get_string_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if is_string_dtype(df[column])]


def save_boxplots(df: pd.DataFrame, columns: list[str], output_path: Path, title: str) -> None:
    rows = int(np.ceil(len(columns) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for index, column in enumerate(columns):
        df[column].plot(kind="box", ax=axes[index])
        axes[index].set_title(column)

    for index in range(len(columns), len(axes)):
        axes[index].set_visible(False)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_hist_comparison(before_df: pd.DataFrame, after_df: pd.DataFrame, output_path: Path) -> None:
    columns = ["age", "monthly_charges", "num_services", "support_calls", "satisfaction_score"]
    rows = int(np.ceil(len(columns) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for index, column in enumerate(columns):
        before_df[column].hist(
            ax=axes[index],
            bins=25,
            alpha=0.55,
            label="Before",
            color="#3498db",
            edgecolor="white",
        )
        after_df[column].hist(
            ax=axes[index],
            bins=25,
            alpha=0.45,
            label="After",
            color="#e74c3c",
            edgecolor="white",
        )
        axes[index].set_title(column)
        axes[index].legend()

    for index in range(len(columns), len(axes)):
        axes[index].set_visible(False)

    fig.suptitle("Step 4: Distribution comparison before and after outlier processing", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Step 5: Correlation matrix after encoding")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_new_features_plot(df: pd.DataFrame, output_path: Path) -> None:
    feature_columns = [
        "charges_per_service",
        "support_calls_per_year",
        "tenure_years",
        "service_intensity",
        "log_monthly_charges",
    ]
    rows = int(np.ceil(len(feature_columns) / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for index, column in enumerate(feature_columns):
        sns.histplot(data=df, x=column, hue=TARGET_COLUMN, bins=25, ax=axes[index], element="step")
        axes[index].set_title(column)

    for index in range(len(feature_columns), len(axes)):
        axes[index].set_visible(False)

    fig.suptitle("Step 6: Engineered feature distributions by churn", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def clip_iqr(df: pd.DataFrame, column: str) -> None:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    before = int(((df[column] < lower) | (df[column] > upper)).sum())
    df[column] = df[column].clip(lower=lower, upper=upper)
    print(f"  {column}: clipped {before} values to [{lower:.2f}, {upper:.2f}]")


def apply_domain_outlier_rules(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result.loc[(result["age"] < 18) | (result["age"] > 100), "age"] = np.nan
    result["age"] = result["age"].fillna(result["age"].median())

    result.loc[result["monthly_charges"] < 0, "monthly_charges"] = np.nan
    result["monthly_charges"] = result["monthly_charges"].fillna(result["monthly_charges"].median())

    result.loc[(result["num_services"] < 0) | (result["num_services"] > 12), "num_services"] = np.nan
    result["num_services"] = result["num_services"].fillna(result["num_services"].median())

    result.loc[
        (result["satisfaction_score"] < 1) | (result["satisfaction_score"] > 5),
        "satisfaction_score",
    ] = np.nan
    result["satisfaction_score"] = result["satisfaction_score"].fillna(
        result["satisfaction_score"].median()
    )

    result.loc[result["support_calls"] < 0, "support_calls"] = np.nan
    result["support_calls"] = result["support_calls"].fillna(result["support_calls"].median())

    result.loc[result["tenure_months"] <= 0, "tenure_months"] = np.nan
    result["tenure_months"] = result["tenure_months"].fillna(result["tenure_months"].median())

    return result


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["gender_male"] = result["gender"].map({"Female": 0, "Male": 1}).astype(int)
    result["contract_type_ord"] = result["contract_type"].map(
        {"Month": 0, "One year": 1, "Two year": 2}
    )

    result = result.drop(columns=["gender", "contract_type"])
    result = pd.get_dummies(
        result,
        columns=["region", "payment_method"],
        prefix=["region", "payment"],
        drop_first=True,
        dtype=int,
    )

    return result


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["tenure_years"] = result["tenure_months"] / 12
    result["charges_per_service"] = result["monthly_charges"] / (result["num_services"] + 1)
    result["support_calls_per_year"] = result["support_calls"] / (result["tenure_years"] + 1)
    result["service_intensity"] = result["num_services"] / (result["tenure_months"] + 1)
    result["satisfaction_per_charge"] = result["satisfaction_score"] / (
        result["monthly_charges"] + 1
    )
    result["log_monthly_charges"] = np.log1p(result["monthly_charges"])
    result["is_long_tenure"] = (result["tenure_months"] > 24).astype(int)
    result["has_many_support_calls"] = (result["support_calls"] >= 5).astype(int)
    result["is_low_satisfaction"] = (result["satisfaction_score"] <= 2).astype(int)

    return result


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}. Run lab_1/lab1_steps_1_3.py first."
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("LAB 2: outliers, encoding, feature engineering")
    print("Variant 1: telecom customer churn")

    df = pd.read_csv(INPUT_PATH)
    assert df.isna().sum().sum() == 0, "Missing values remain from Lab 1."
    assert df.duplicated(subset=[ID_COLUMN]).sum() == 0, "Duplicate customer_id values remain."

    print(f"\nLoaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"String columns before encoding: {get_string_columns(df)}")

    outlier_columns = [
        "age",
        "tenure_months",
        "monthly_charges",
        "num_services",
        "support_calls",
        "satisfaction_score",
    ]
    save_boxplots(
        df,
        outlier_columns,
        ARTIFACTS_DIR / "step4_boxplots_before.png",
        "Step 4: Boxplots before outlier processing",
    )

    df_before_outliers = df.copy()
    print("\nSTEP 4. OUTLIER PROCESSING")
    df = apply_domain_outlier_rules(df)
    for column in ["age", "monthly_charges", "num_services", "support_calls"]:
        clip_iqr(df, column)

    integer_like_columns = ["tenure_months", "num_services", "support_calls"]
    for column in integer_like_columns:
        df[column] = df[column].round()

    save_boxplots(
        df,
        outlier_columns,
        ARTIFACTS_DIR / "step4_boxplots_after.png",
        "Step 4: Boxplots after outlier processing",
    )
    save_hist_comparison(df_before_outliers, df, ARTIFACTS_DIR / "step4_hist_compare.png")

    print("\nSTEP 5. CATEGORICAL ENCODING")
    df = encode_categorical_features(df)
    print(f"Columns after encoding: {df.shape[1]}")
    print(f"String columns after encoding: {get_string_columns(df)}")
    save_correlation_heatmap(df, ARTIFACTS_DIR / "step5_correlation.png")

    print("\nSTEP 6. FEATURE ENGINEERING")
    df = add_engineered_features(df)
    save_new_features_plot(df, ARTIFACTS_DIR / "step6_new_features.png")

    assert df.isna().sum().sum() == 0, "Missing values remain after Lab 2."
    assert not df.isin([np.inf, -np.inf]).any().any(), "Infinite values were created."
    assert len(get_string_columns(df)) == 0, "Non-numeric string columns remain."

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nLAB 2 REPORT")
    print(f"Final size: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Domain rules corrected impossible telecom values.")
    print("Remaining numeric extremes were clipped with the IQR rule.")
    print("Categorical columns were encoded with binary, ordinal, and one-hot encoding.")
    print("New behavioral and monetary features were added.")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
