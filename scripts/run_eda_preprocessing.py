from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import (
    class_distribution,
    encode_labels,
    load_raw_dataset,
    preprocess_dataset,
    stratified_train_val_test_split,
)
from src.visualization import save_class_distribution, save_text_length_distribution, save_top_terms_chart, save_wordcloud_grid, top_terms_by_class


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Combined_Data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
SCREENSHOTS_DIR = RESULTS_DIR / "screenshots"
MODELS_DIR = ROOT / "models"


EXPECTED_AUDIT = {
    "raw_rows": 53043,
    "null_statements": 362,
    "rows_after_null_removal": 52681,
    "empty_after_cleaning": 139,
    "final_modelling_rows": 52542,
}


def assert_expected_counts(audit: dict) -> None:
    mismatches = {
        key: {"expected": expected, "actual": audit.get(key)}
        for key, expected in EXPECTED_AUDIT.items()
        if audit.get(key) != expected
    }
    if mismatches:
        raise AssertionError(f"Preprocessing counts do not match the verified master plan: {mismatches}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataset(DATA_PATH)
    non_null_distribution = class_distribution(raw_df.dropna(subset=["statement"]))
    final_df, audit = preprocess_dataset(raw_df)
    assert_expected_counts(audit)

    final_distribution = class_distribution(final_df)
    encoded_df, label_encoder = encode_labels(final_df)
    train_df, val_df, test_df = stratified_train_val_test_split(encoded_df)

    split_audit = {
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
    }
    if split_audit != {"train_rows": 36779, "validation_rows": 7881, "test_rows": 7882}:
        raise AssertionError(f"Split sizes do not match master plan: {split_audit}")

    final_df.to_csv(PROCESSED_DIR / "cleaned_full.csv", index=False)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.joblib")

    non_null_distribution.to_csv(TABLES_DIR / "class_distribution_after_null_removal.csv", index=False)
    final_distribution.to_csv(TABLES_DIR / "class_distribution_final_cleaned.csv", index=False)
    top_terms = top_terms_by_class(final_df, top_n=10)
    top_terms.to_csv(TABLES_DIR / "top_terms_per_class.csv", index=False)

    with (TABLES_DIR / "dataset_audit.json").open("w", encoding="utf-8") as f:
        json.dump({**audit, **split_audit}, f, indent=2)

    save_class_distribution(final_distribution, SCREENSHOTS_DIR / "class_distribution.png")
    save_text_length_distribution(final_df, SCREENSHOTS_DIR / "text_length_distribution.png")
    save_wordcloud_grid(final_df, SCREENSHOTS_DIR / "wordcloud_all_classes.png")
    save_top_terms_chart(top_terms, SCREENSHOTS_DIR / "top_opinion_targets_per_class.png")

    print(json.dumps({**audit, **split_audit}, indent=2))


if __name__ == "__main__":
    main()
