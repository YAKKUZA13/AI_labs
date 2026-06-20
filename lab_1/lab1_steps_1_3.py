"""
Laboratory work 1. Loading, duplicate processing, and missing-value analysis.

Variant 1: telecom customer churn.
Input: artifacts/customer_churn.csv
Output: artifacts/data_after_step3.csv and diagnostic plots.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_string_dtype
from sklearn.impute import KNNImputer


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
INPUT_PATH = ARTIFACTS_DIR / "customer_churn.csv"
OUTPUT_PATH = ARTIFACTS_DIR / "data_after_step3.csv"
ID_COLUMN = "customer_id"
TARGET_COLUMN = "churn"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150


def get_string_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if is_string_dtype(df[column])]


def save_numeric_histograms(df: pd.DataFrame, output_path: Path) -> None:
    numeric_columns = [col for col in df.select_dtypes(include="number").columns if col != ID_COLUMN]
    axes_count = max(len(numeric_columns), 1)
    rows = int(np.ceil(axes_count / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for index, column in enumerate(numeric_columns):
        df[column].hist(ax=axes[index], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        axes[index].set_title(column)
        axes[index].set_xlabel("Value")
        axes[index].set_ylabel("Count")

    for index in range(len(numeric_columns), len(axes)):
        axes[index].set_visible(False)

    fig.suptitle("Step 1: Numeric feature distributions before processing", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_categorical_bars(df: pd.DataFrame, output_path: Path) -> None:
    categorical_columns = get_string_columns(df)
    axes_count = max(len(categorical_columns), 1)
    rows = int(np.ceil(axes_count / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
    axes = np.array(axes).reshape(-1)

    for index, column in enumerate(categorical_columns):
        df[column].value_counts(dropna=False).plot.bar(
            ax=axes[index],
            color="coral",
            edgecolor="black",
        )
        axes[index].set_title(column)
        axes[index].tick_params(axis="x", rotation=35)

    for index in range(len(categorical_columns), len(axes)):
        axes[index].set_visible(False)

    fig.suptitle("Step 1: Categorical feature distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_duplicate_plot(complete_duplicates: int, id_duplicates: int, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        ["Complete duplicates", "customer_id duplicates"],
        [complete_duplicates, id_duplicates],
        color=["#3498db", "#e67e22"],
        edgecolor="black",
    )
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            str(int(bar.get_height())),
            ha="center",
            fontweight="bold",
        )
    plt.title("Step 2: Duplicate records")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_missing_heatmap(df: pd.DataFrame, output_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_missing_bars(missing_counts: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    missing_counts[missing_counts > 0].sort_values(ascending=False).plot.bar(
        color="#9b59b6",
        edgecolor="black",
    )
    plt.title("Step 3: Missing values by column")
    plt.ylabel("Missing values")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_distribution_comparison(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    output_path: Path,
) -> None:
    columns = ["age", "tenure_months", "support_calls", "satisfaction_score"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.reshape(-1)

    for index, column in enumerate(columns):
        before_df[column].dropna().hist(
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

    fig.suptitle("Step 3: Numeric distributions before and after imputation", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def fill_categorical_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    payment_by_contract = result.groupby("contract_type")["payment_method"].transform(
        lambda values: values.mode().iloc[0] if not values.mode().empty else np.nan
    )
    result["payment_method"] = result["payment_method"].fillna(payment_by_contract)
    result["payment_method"] = result["payment_method"].fillna(result["payment_method"].mode().iloc[0])

    region_by_payment = result.groupby("payment_method")["region"].transform(
        lambda values: values.mode().iloc[0] if not values.mode().empty else np.nan
    )
    result["region"] = result["region"].fillna(region_by_payment)
    result["region"] = result["region"].fillna(result["region"].mode().iloc[0])

    for column in get_string_columns(result):
        if result[column].isna().any():
            result[column] = result[column].fillna(result[column].mode().iloc[0])

    return result


def impute_numeric_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    numeric_columns = [
        column
        for column in result.select_dtypes(include="number").columns
        if column not in [ID_COLUMN, TARGET_COLUMN]
    ]
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    imputed = imputer.fit_transform(result[numeric_columns])
    result[numeric_columns] = pd.DataFrame(imputed, columns=numeric_columns, index=result.index)

    integer_like_columns = ["tenure_months", "num_services", "support_calls"]
    for column in integer_like_columns:
        result[column] = result[column].round()

    return result


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}. Run lab_1/generate_data.py first."
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("LAB 1: loading, duplicates, missing values")
    print("Variant 1: telecom customer churn")

    df = pd.read_csv(INPUT_PATH)
    df_original = df.copy()

    print("\nSTEP 1. LOAD AND INSPECT")
    print(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nColumn types:")
    print(df.dtypes.to_string())
    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print("\nNumeric statistics:")
    print(df.describe().round(2).to_string())

    save_numeric_histograms(df, ARTIFACTS_DIR / "step1_hist_numeric_before.png")
    save_categorical_bars(df, ARTIFACTS_DIR / "step1_categorical.png")

    print("\nSTEP 2. DUPLICATE PROCESSING")
    complete_duplicates = int(df.duplicated().sum())
    id_duplicates = int(df.duplicated(subset=[ID_COLUMN]).sum())
    print(f"Complete duplicates: {complete_duplicates}")
    print(f"Duplicate {ID_COLUMN}: {id_duplicates}")

    save_duplicate_plot(complete_duplicates, id_duplicates, ARTIFACTS_DIR / "step2_duplicates.png")
    df = df.drop_duplicates(subset=[ID_COLUMN], keep="first").reset_index(drop=True)
    print(f"Rows after duplicate removal: {df.shape[0]}")

    print("\nSTEP 3. MISSING VALUE ANALYSIS AND PROCESSING")
    missing_before = df.isna().sum()
    print("Missing before processing:")
    print(missing_before[missing_before > 0].to_string())

    save_missing_heatmap(
        df,
        ARTIFACTS_DIR / "step3_missing_heatmap_before.png",
        "Step 3: Missing values before processing",
    )
    save_missing_bars(missing_before, ARTIFACTS_DIR / "step3_missing_bars.png")

    df_before_imputation = df.copy()
    df = fill_categorical_missing_values(df)
    df = impute_numeric_missing_values(df)

    missing_after = int(df.isna().sum().sum())
    print(f"Missing after processing: {missing_after}")

    save_missing_heatmap(
        df,
        ARTIFACTS_DIR / "step3_missing_heatmap_after.png",
        "Step 3: Missing values after processing",
    )
    save_distribution_comparison(
        df_before_imputation,
        df,
        ARTIFACTS_DIR / "step3_distributions_compare.png",
    )

    assert df.duplicated(subset=[ID_COLUMN]).sum() == 0, "Duplicate customer_id values remain."
    assert df.isna().sum().sum() == 0, "Missing values remain after processing."
    assert df.shape[0] < df_original.shape[0], "Duplicate removal did not reduce row count."

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nLAB 1 REPORT")
    print(f"Initial size: {df_original.shape[0]} rows, {df_original.shape[1]} columns")
    print(f"Final size: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Duplicates were removed by customer_id.")
    print("Numeric missing values were filled with KNNImputer.")
    print("Categorical missing values were filled with grouped modes and fallback modes.")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
