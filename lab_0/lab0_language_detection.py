"""
Laboratory work 0. Hugging Face language detection.

Variant 1: determine the language of 10 multilingual sentences.
Output: artifacts/language_detection_results.csv and artifacts/language_detection_summary.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
RESULTS_CSV_PATH = ARTIFACTS_DIR / "language_detection_results.csv"
SUMMARY_JSON_PATH = ARTIFACTS_DIR / "language_detection_summary.json"
MODEL_ID = "alexneakameni/language_detection"

LANGUAGE_NAMES = {
    "ru": "Русский",
    "en": "Английский",
    "de": "Немецкий",
    "fr": "Французский",
    "zh": "Китайский",
}

MODEL_LABEL_TO_LANGUAGE = {
    "rus_Cyrl": "ru",
    "eng_Latn": "en",
    "deu_Latn": "de",
    "fra_Latn": "fr",
    "zho_Hans": "zh",
    "zho_Hant": "zh",
}

TEST_SENTENCES = [
    {
        "id": 1,
        "expected_label": "ru",
        "text": "Сегодня студенты выполняют лабораторную работу по искусственному интеллекту.",
    },
    {
        "id": 2,
        "expected_label": "ru",
        "text": "Модель должна определить язык короткого предложения без подсказок.",
    },
    {
        "id": 3,
        "expected_label": "en",
        "text": "The neural network analyzes the sentence and predicts its language.",
    },
    {
        "id": 4,
        "expected_label": "en",
        "text": "This example checks whether the model works correctly on English text.",
    },
    {
        "id": 5,
        "expected_label": "de",
        "text": "Die Studierenden testen ein Modell zur automatischen Spracherkennung.",
    },
    {
        "id": 6,
        "expected_label": "de",
        "text": "Dieses Beispiel enthält einen deutschen Satz mit typischer Wortstellung.",
    },
    {
        "id": 7,
        "expected_label": "fr",
        "text": "Le modèle doit reconnaître correctement la langue française.",
    },
    {
        "id": 8,
        "expected_label": "fr",
        "text": "Cette phrase est utilisée pour vérifier la qualité de la prédiction.",
    },
    {
        "id": 9,
        "expected_label": "zh",
        "text": "这个句子用于检查模型是否能够识别中文。",
    },
    {
        "id": 10,
        "expected_label": "zh",
        "text": "人工智能模型应该正确地判断这段文字的语言。",
    },
]


def normalize_prediction(raw_prediction: Any) -> dict[str, Any]:
    """Return a single top prediction from pipeline output across Transformers versions."""
    if isinstance(raw_prediction, list) and raw_prediction and isinstance(raw_prediction[0], list):
        return raw_prediction[0][0]
    if isinstance(raw_prediction, list) and raw_prediction:
        return raw_prediction[0]
    if isinstance(raw_prediction, dict):
        return raw_prediction
    raise ValueError(f"Unexpected prediction format: {raw_prediction!r}")


def build_detector():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, top_k=1)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print("LAB 0: Hugging Face language detection")
    print(f"Model: {MODEL_ID}")

    started_at = time.perf_counter()
    detector = build_detector()
    model_load_seconds = time.perf_counter() - started_at

    rows = []
    inference_started_at = time.perf_counter()
    for item in TEST_SENTENCES:
        raw_prediction = detector(item["text"], truncation=True)
        prediction = normalize_prediction(raw_prediction)
        predicted_raw_label = str(prediction["label"])
        predicted_label = MODEL_LABEL_TO_LANGUAGE.get(predicted_raw_label, predicted_raw_label)
        score = float(prediction["score"])
        is_correct = predicted_label == item["expected_label"]

        rows.append(
            {
                "id": item["id"],
                "text": item["text"],
                "expected_label": item["expected_label"],
                "expected_language": LANGUAGE_NAMES[item["expected_label"]],
                "predicted_raw_label": predicted_raw_label,
                "predicted_label": predicted_label,
                "predicted_language": LANGUAGE_NAMES.get(predicted_label, predicted_label),
                "score": round(score, 6),
                "is_correct": is_correct,
            }
        )

    inference_seconds = time.perf_counter() - inference_started_at
    results = pd.DataFrame(rows)
    accuracy = float(results["is_correct"].mean())

    results.to_csv(RESULTS_CSV_PATH, index=False, encoding="utf-8")
    summary = {
        "model_id": MODEL_ID,
        "task": "language detection",
        "total_sentences": int(len(results)),
        "correct_predictions": int(results["is_correct"].sum()),
        "accuracy": round(accuracy, 4),
        "model_load_seconds": round(model_load_seconds, 4),
        "inference_seconds": round(inference_seconds, 4),
        "supported_test_languages": LANGUAGE_NAMES,
    }
    with SUMMARY_JSON_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nResults:")
    print(results.to_string(index=False))
    print("\nSummary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved: {RESULTS_CSV_PATH}")
    print(f"Saved: {SUMMARY_JSON_PATH}")

    assert len(results) == 10, "The assignment requires exactly 10 test sentences."
    assert accuracy >= 0.8, "Language detection quality is too low for the selected examples."


if __name__ == "__main__":
    main()
