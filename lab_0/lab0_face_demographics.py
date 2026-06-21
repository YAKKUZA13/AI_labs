"""
Laboratory work 0. Hugging Face face demographics analysis.

Variant 9: determine gender and age group from 10 face photos.
Output: artifacts/face_analysis_results.csv and artifacts/face_analysis_summary.json
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from transformers import pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
TEST_IMAGES_DIR = SCRIPT_DIR / "test_images"
MANIFEST_PATH = SCRIPT_DIR / "test_manifest.csv"
RESULTS_CSV_PATH = ARTIFACTS_DIR / "face_analysis_results.csv"
SUMMARY_JSON_PATH = ARTIFACTS_DIR / "face_analysis_summary.json"

GENDER_MODEL_ID = "dima806/fairface_gender_image_detection"
AGE_MODEL_ID = "dima806/fairface_age_image_detection"

VALID_AGE_GROUPS = {
    "0-2",
    "3-9",
    "10-19",
    "20-29",
    "30-39",
    "40-49",
    "50-59",
    "60-69",
    "more than 70",
}


def normalize_prediction(raw_prediction: Any) -> dict[str, Any]:
    """Return a single top prediction from pipeline output across Transformers versions."""
    if isinstance(raw_prediction, list) and raw_prediction and isinstance(raw_prediction[0], list):
        return raw_prediction[0][0]
    if isinstance(raw_prediction, list) and raw_prediction:
        return raw_prediction[0]
    if isinstance(raw_prediction, dict):
        return raw_prediction
    raise ValueError(f"Unexpected prediction format: {raw_prediction!r}")


def normalize_label(raw_label: str) -> str:
    label = raw_label.strip()
    label = re.sub(r"^LABEL_\d+$", "", label).strip()
    return re.sub(r"\s+", " ", label)


def normalize_gender(raw_label: str) -> str:
    label = normalize_label(raw_label).lower()
    if label in {"female", "f", "woman", "women"}:
        return "Female"
    if label in {"male", "m", "man", "men"}:
        return "Male"
    raise ValueError(f"Unsupported gender label: {raw_label!r}")


def normalize_age_group(raw_label: str) -> str:
    label = normalize_label(raw_label).lower()
    replacements = {
        "more_than_70": "more than 70",
        "more-than-70": "more than 70",
        "70+": "more than 70",
        "70 plus": "more than 70",
    }
    label = replacements.get(label, label)

    for age_group in VALID_AGE_GROUPS:
        if label == age_group.lower():
            return age_group

    raise ValueError(f"Unsupported age group label: {raw_label!r}")


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing test manifest: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    required_columns = {"id", "image_file", "expected_gender", "expected_age_group"}
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    if len(manifest) != 10:
        raise ValueError("The assignment requires exactly 10 test photos.")

    for _, row in manifest.iterrows():
        image_path = TEST_IMAGES_DIR / str(row["image_file"])
        if not image_path.exists():
            raise FileNotFoundError(f"Missing test image: {image_path}")
        normalize_gender(str(row["expected_gender"]))
        normalize_age_group(str(row["expected_age_group"]))

    return manifest


def build_classifier(model_id: str):
    return pipeline("image-classification", model=model_id, top_k=1)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()

    print("LAB 0: Hugging Face face demographics analysis")
    print(f"Gender model: {GENDER_MODEL_ID}")
    print(f"Age model: {AGE_MODEL_ID}")

    started_at = time.perf_counter()
    gender_classifier = build_classifier(GENDER_MODEL_ID)
    age_classifier = build_classifier(AGE_MODEL_ID)
    model_load_seconds = time.perf_counter() - started_at

    rows = []
    inference_started_at = time.perf_counter()
    for _, item in manifest.iterrows():
        image_path = TEST_IMAGES_DIR / str(item["image_file"])
        image = Image.open(image_path).convert("RGB")

        gender_prediction = normalize_prediction(gender_classifier(image))
        age_prediction = normalize_prediction(age_classifier(image))

        predicted_gender = normalize_gender(str(gender_prediction["label"]))
        predicted_age_group = normalize_age_group(str(age_prediction["label"]))
        expected_gender = normalize_gender(str(item["expected_gender"]))
        expected_age_group = normalize_age_group(str(item["expected_age_group"]))

        rows.append(
            {
                "id": int(item["id"]),
                "image_file": str(item["image_file"]),
                "expected_gender": expected_gender,
                "expected_age_group": expected_age_group,
                "predicted_gender": predicted_gender,
                "predicted_age_group": predicted_age_group,
                "gender_score": round(float(gender_prediction["score"]), 6),
                "age_score": round(float(age_prediction["score"]), 6),
                "is_gender_correct": predicted_gender == expected_gender,
                "is_age_correct": predicted_age_group == expected_age_group,
            }
        )

    inference_seconds = time.perf_counter() - inference_started_at
    results = pd.DataFrame(rows)
    gender_accuracy = float(results["is_gender_correct"].mean())
    age_accuracy = float(results["is_age_correct"].mean())

    results.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")
    summary = {
        "gender_model_id": GENDER_MODEL_ID,
        "age_model_id": AGE_MODEL_ID,
        "task": "face gender and age group detection",
        "total_photos": int(len(results)),
        "gender_correct_predictions": int(results["is_gender_correct"].sum()),
        "age_correct_predictions": int(results["is_age_correct"].sum()),
        "gender_accuracy": round(gender_accuracy, 4),
        "age_accuracy": round(age_accuracy, 4),
        "model_load_seconds": round(model_load_seconds, 4),
        "inference_seconds": round(inference_seconds, 4),
        "test_manifest": str(MANIFEST_PATH.relative_to(SCRIPT_DIR)),
        "test_images_dir": str(TEST_IMAGES_DIR.relative_to(SCRIPT_DIR)),
    }
    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nResults:")
    print(results.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved: {RESULTS_CSV_PATH}")
    print(f"Saved: {SUMMARY_JSON_PATH}")

    assert len(results) == 10, "The assignment requires exactly 10 test photos."
    assert gender_accuracy >= 0.8, "Gender detection quality is too low for the selected photos."
    assert age_accuracy >= 0.5, "Age group detection quality is too low for the selected photos."


if __name__ == "__main__":
    main()
