"""One-time helper to export 10 diverse FairFace samples for lab 0 variant 9."""

from __future__ import annotations

import csv
import pickle
from io import BytesIO
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
TEST_IMAGES_DIR = SCRIPT_DIR / "test_images"
MANIFEST_PATH = SCRIPT_DIR / "test_manifest.csv"

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


def normalize_age_group(raw_age: str) -> str:
    age_group = str(raw_age).strip()
    if age_group not in VALID_AGE_GROUPS:
        raise ValueError(f"Unsupported age group value: {raw_age!r}")
    return age_group


def normalize_gender(raw_gender: str | int) -> str:
    gender = str(raw_gender).strip().lower()
    if gender in {"female", "f", "1"}:
        return "Female"
    if gender in {"male", "m", "0"}:
        return "Male"
    raise ValueError(f"Unsupported gender value: {raw_gender!r}")


def decode_image(raw_image: bytes | Image.Image) -> Image.Image:
    if isinstance(raw_image, Image.Image):
        return raw_image.convert("RGB")
    if isinstance(raw_image, (bytes, bytearray)):
        return Image.open(BytesIO(raw_image)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(raw_image)!r}")


def load_fairface_split() -> list[dict]:
    split_path = hf_hub_download("nateraw/fairface", "val.pt", repo_type="dataset")
    with open(split_path, "rb") as file:
        data = pickle.load(file)

    if not isinstance(data, list):
        raise RuntimeError(f"Unexpected FairFace split type: {type(data)!r}")

    return data


def main() -> None:
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    dataset_rows = load_fairface_split()

    selected_rows: list[dict[str, str | int]] = []
    covered_groups: set[str] = set()
    seen_signatures: set[tuple[str, str]] = set()

    for row in dataset_rows:
        age_group = normalize_age_group(row["age"])
        gender = normalize_gender(row["gender"])
        signature = (gender, age_group)

        should_select = False
        if len(selected_rows) < 9 and age_group not in covered_groups:
            should_select = True
        elif len(selected_rows) == 9 and signature not in seen_signatures:
            should_select = True

        if not should_select:
            continue

        image_id = len(selected_rows) + 1
        image_name = f"face_{image_id:02d}.jpg"
        image_path = TEST_IMAGES_DIR / image_name
        decode_image(row["img_bytes"]).save(image_path, format="JPEG", quality=95)

        selected_rows.append(
            {
                "id": image_id,
                "image_file": image_name,
                "expected_gender": gender,
                "expected_age_group": age_group,
            }
        )
        covered_groups.add(age_group)
        seen_signatures.add(signature)

        if len(selected_rows) == 10:
            break

    if len(selected_rows) < 10:
        raise RuntimeError("Could not collect 10 diverse FairFace samples.")

    with MANIFEST_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["id", "image_file", "expected_gender", "expected_age_group"],
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"Saved {len(selected_rows)} images to {TEST_IMAGES_DIR}")
    print(f"Saved manifest to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
