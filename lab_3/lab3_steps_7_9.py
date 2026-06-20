"""
Laboratory work 3. Train/test split, scaling, and final validation.

Variant 1: telecom customer churn.
Input: ../lab_2/artifacts/data_after_step6.csv
Output: train/test CSV files and diagnostic plots.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
INPUT_PATH = ROOT_DIR / "lab_2" / "artifacts" / "data_after_step6.csv"
TARGET_COLUMN = "churn"
ID_COLUMN = "customer_id"
RANDOM_STATE = 42

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["figure.dpi"] = 150


def get_string_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if is_string_dtype(df[column])]


def is_binary_zero_one(series: pd.Series) -> bool:
    values = set(series.dropna().unique().tolist())
    return bool(values) and values.issubset({0, 1})


def get_columns_to_scale(X_train: pd.DataFrame) -> list[str]:
    return [column for column in X_train.columns if not is_binary_zero_one(X_train[column])]


def save_split_plot(y_train: pd.Series, y_test: pd.Series, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    sizes = [len(y_train), len(y_test)]
    bars = axes[0].bar(["Train", "Test"], sizes, color=["#3498db", "#e74c3c"], edgecolor="black")
    for bar in bars:
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            str(int(bar.get_height())),
            ha="center",
            fontweight="bold",
        )
    axes[0].set_title("Train/test sizes")
    axes[0].set_ylabel("Rows")

    class_balance = pd.DataFrame(
        {
            "train": y_train.value_counts(normalize=True).sort_index(),
            "test": y_test.value_counts(normalize=True).sort_index(),
        }
    )
    class_balance.plot.bar(ax=axes[1], color=["#3498db", "#e74c3c"], edgecolor="black")
    axes[1].set_title("Class proportions after stratification")
    axes[1].set_xlabel("churn")
    axes[1].set_ylabel("Share")
    axes[1].tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_normalization_compare(
    x_train_before: pd.DataFrame,
    x_train_after: pd.DataFrame,
    output_path: Path,
) -> None:
    columns = ["age", "monthly_charges", "tenure_months", "support_calls"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.reshape(-1)

    for index, column in enumerate(columns):
        x_train_before[column].hist(
            ax=axes[index],
            bins=25,
            alpha=0.55,
            label="Before",
            color="#3498db",
            edgecolor="white",
        )
        x_train_after[column].hist(
            ax=axes[index],
            bins=25,
            alpha=0.45,
            label="After",
            color="#e74c3c",
            edgecolor="white",
        )
        axes[index].set_title(column)
        axes[index].legend()

    fig.suptitle("Step 8: Train distributions before and after scaling", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_normalized_boxplots(x_train: pd.DataFrame, output_path: Path) -> None:
    selected_columns = [
        "age",
        "tenure_months",
        "monthly_charges",
        "num_services",
        "support_calls",
        "satisfaction_score",
    ]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=x_train[selected_columns])
    plt.title("Step 8: Boxplots after StandardScaler")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_final_correlation(x_train: pd.DataFrame, y_train: pd.Series, output_path: Path) -> None:
    analysis_df = x_train.copy()
    analysis_df[TARGET_COLUMN] = y_train.values
    corr = analysis_df.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.2)
    plt.title("Step 9: Final correlation matrix on train data")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def save_top_features(x_train: pd.DataFrame, y_train: pd.Series, output_path: Path) -> None:
    correlations = x_train.corrwith(y_train).abs().sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    correlations.sort_values().plot.barh(color="#2ecc71", edgecolor="black")
    plt.title("Step 9: Top features by absolute correlation with churn")
    plt.xlabel("Absolute correlation")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}. Run lab_2/lab2_steps_4_6.py first."
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("LAB 3: train/test split, scaling, final validation")
    print("Variant 1: telecom customer churn")

    df = pd.read_csv(INPUT_PATH)
    assert df.isna().sum().sum() == 0, "Missing values remain before Lab 3."
    assert not df.isin([np.inf, -np.inf]).any().any(), "Infinite values exist before Lab 3."
    assert len(get_string_columns(df)) == 0, "Non-numeric string columns remain."
    assert TARGET_COLUMN in df.columns, f"Target column {TARGET_COLUMN} is missing."

    print(f"\nLoaded: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\nSTEP 7. TRAIN/TEST SPLIT")
    y = df[TARGET_COLUMN]
    customer_ids = df[ID_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN, ID_COLUMN])

    X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(
        X,
        y,
        customer_ids,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Train rows: {X_train.shape[0]}")
    print(f"Test rows: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Train class balance: {y_train.value_counts(normalize=True).round(3).to_dict()}")
    print(f"Test class balance: {y_test.value_counts(normalize=True).round(3).to_dict()}")

    save_split_plot(y_train, y_test, ARTIFACTS_DIR / "step7_split.png")

    print("\nSTEP 8. STANDARDIZATION")
    X_train_before_scaling = X_train.copy()
    columns_to_scale = get_columns_to_scale(X_train)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_test_scaled[columns_to_scale] = scaler.transform(X_test[columns_to_scale])

    train_mean_max_abs = float(X_train_scaled[columns_to_scale].mean().abs().max())
    train_std_mean = float(X_train_scaled[columns_to_scale].std(ddof=0).mean())
    print(f"Scaled columns: {len(columns_to_scale)}")
    print(f"Max absolute train mean after scaling: {train_mean_max_abs:.10f}")
    print(f"Mean train std after scaling: {train_std_mean:.4f}")

    save_normalization_compare(
        X_train_before_scaling,
        X_train_scaled,
        ARTIFACTS_DIR / "step8_normalization_compare.png",
    )
    save_normalized_boxplots(X_train_scaled, ARTIFACTS_DIR / "step8_boxplots_normalized.png")

    print("\nSTEP 9. FINAL VALIDATION AND SAVING")
    assert X_train_scaled.isna().sum().sum() == 0, "Missing values in X_train."
    assert X_test_scaled.isna().sum().sum() == 0, "Missing values in X_test."
    assert not X_train_scaled.isin([np.inf, -np.inf]).any().any(), "Infinite values in X_train."
    assert not X_test_scaled.isin([np.inf, -np.inf]).any().any(), "Infinite values in X_test."
    assert list(X_train_scaled.columns) == list(X_test_scaled.columns), "Feature columns differ."
    balance_tolerance = max(0.02, 1 / len(y_test))
    assert train_mean_max_abs < 1e-9, "Scaler was not fitted correctly on train data."
    assert abs(y_train.mean() - y_test.mean()) <= balance_tolerance, (
        "Stratification changed class balance too much."
    )

    save_final_correlation(X_train_scaled, y_train, ARTIFACTS_DIR / "step9_final_correlation.png")
    save_top_features(X_train_scaled, y_train, ARTIFACTS_DIR / "step9_top_features.png")

    X_train_scaled.sort_index().to_csv(ARTIFACTS_DIR / "X_train.csv", index=False)
    X_test_scaled.sort_index().to_csv(ARTIFACTS_DIR / "X_test.csv", index=False)
    y_train.sort_index().to_csv(ARTIFACTS_DIR / "y_train.csv", index=False)
    y_test.sort_index().to_csv(ARTIFACTS_DIR / "y_test.csv", index=False)
    train_ids.sort_index().to_csv(ARTIFACTS_DIR / "train_ids.csv", index=False)
    test_ids.sort_index().to_csv(ARTIFACTS_DIR / "test_ids.csv", index=False)

    metadata = {
        "target_column": TARGET_COLUMN,
        "id_column": ID_COLUMN,
        "feature_columns": X_train_scaled.columns.tolist(),
        "scaled_columns": columns_to_scale,
        "not_scaled_columns": [
            column for column in X_train_scaled.columns if column not in columns_to_scale
        ],
        "split": {
            "test_size": 0.2,
            "random_state": RANDOM_STATE,
            "stratify": TARGET_COLUMN,
            "train_rows": int(X_train_scaled.shape[0]),
            "test_rows": int(X_test_scaled.shape[0]),
        },
        "standard_scaler": {
            "mean": dict(zip(columns_to_scale, scaler.mean_.tolist())),
            "scale": dict(zip(columns_to_scale, scaler.scale_.tolist())),
            "var": dict(zip(columns_to_scale, scaler.var_.tolist())),
        },
    }
    with (ARTIFACTS_DIR / "preprocessing_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("\nLAB 3 REPORT")
    print(f"Final train shape: {X_train_scaled.shape}")
    print(f"Final test shape: {X_test_scaled.shape}")
    print("Target: churn")
    print("Split: 80/20 with stratification")
    print("Scaling: StandardScaler fitted on train and applied to test")
    print("Saved IDs and preprocessing metadata for reproducibility")
    print(f"Saved artifacts to: {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
