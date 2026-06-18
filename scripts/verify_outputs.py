from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
SCREENSHOTS_DIR = RESULTS_DIR / "screenshots"
NOTEBOOK_DIR = ROOT / "notebooks"


EXPECTED_AUDIT = {
    "raw_rows": 53043,
    "null_statements": 362,
    "rows_after_null_removal": 52681,
    "empty_after_cleaning": 138,
    "final_modelling_rows": 52543,
    "train_rows": 36780,
    "validation_rows": 7881,
    "test_rows": 7882,
}

EXPECTED_COUNTS = {
    "Normal": 16210,
    "Depression": 15403,
    "Suicidal": 10648,
    "Anxiety": 3841,
    "Bipolar": 2777,
    "Stress": 2587,
    "Personality disorder": 1077,
}

REQUIRED_SCREENSHOTS = [
    "class_distribution.png",
    "text_length_distribution.png",
    "wordcloud_all_classes.png",
    "top_opinion_targets_per_class.png",
    "confusion_matrix_naive_bayes.png",
    "confusion_matrix_logistic_regression.png",
    "confusion_matrix_linear_svm.png",
    "traditional_ml_macro_f1_comparison.png",
    "vader_sentiment_distribution.png",
    "absa_heatmap.png",
]

REQUIRED_TABLES = [
    "dataset_audit.json",
    "class_distribution_final_cleaned.csv",
    "traditional_ml_results.csv",
    "model_comparison_table.csv",
    "vader_sentiment_summary_by_class.csv",
    "aspect_mentions_by_class.csv",
    "aspect_sentiment_summary.csv",
]


def verify_audit() -> None:
    audit = json.loads((TABLES_DIR / "dataset_audit.json").read_text(encoding="utf-8"))
    for key, expected in EXPECTED_AUDIT.items():
        actual = audit.get(key)
        if actual != expected:
            raise AssertionError(f"Audit mismatch for {key}: expected {expected}, got {actual}")


def verify_class_distribution() -> None:
    distribution = pd.read_csv(TABLES_DIR / "class_distribution_final_cleaned.csv")
    actual_counts = dict(zip(distribution["status"], distribution["count"]))
    if actual_counts != EXPECTED_COUNTS:
        raise AssertionError(f"Class counts mismatch: {actual_counts}")
    if int(distribution["count"].sum()) != EXPECTED_AUDIT["final_modelling_rows"]:
        raise AssertionError("Class distribution does not sum to final modelling rows.")


def verify_model_results() -> None:
    results = pd.read_csv(TABLES_DIR / "traditional_ml_results.csv")
    expected_models = {"Naive Bayes", "Logistic Regression", "Linear SVM"}
    if set(results["model"]) != expected_models:
        raise AssertionError(f"Traditional ML result models mismatch: {set(results['model'])}")
    numeric_cols = ["train_accuracy", "validation_accuracy", "test_accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
    if results[numeric_cols].isna().any().any():
        raise AssertionError("Traditional ML results contain missing numeric values.")
    if not ((results[numeric_cols] >= 0).all().all() and (results[numeric_cols] <= 1).all().all()):
        raise AssertionError("Traditional ML metrics must be within [0, 1].")

    comparison = pd.read_csv(TABLES_DIR / "model_comparison_table.csv")
    expected_order = ["Naive Bayes", "Logistic Regression", "Linear SVM", "CNN", "DistilBERT", "BiLSTM", "GRU", "BERT", "RoBERTa"]
    if comparison["model"].tolist() != expected_order:
        raise AssertionError(f"Unexpected model comparison order: {comparison['model'].tolist()}")
    for column in ["family", "priority", "status", "notes"]:
        if column not in comparison.columns:
            raise AssertionError(f"Model comparison table missing column: {column}")
    traditional_status = dict(zip(comparison["model"], comparison["status"]))
    for model in ["Naive Bayes", "Logistic Regression", "Linear SVM"]:
        if traditional_status.get(model) != "executed":
            raise AssertionError(f"{model} must remain an executed traditional baseline.")
    if set(comparison["status"]) - {"executed", "pending"}:
        raise AssertionError(f"Unexpected model statuses: {set(comparison['status'])}")
    executed_rows = comparison[comparison["status"] == "executed"]
    required_executed_metrics = ["test_accuracy", "macro_f1", "macro_precision", "macro_recall"]
    if executed_rows[required_executed_metrics].isna().any().any():
        raise AssertionError("Executed model rows must contain numeric test metrics.")
    if not ((executed_rows[required_executed_metrics] >= 0).all().all() and (executed_rows[required_executed_metrics] <= 1).all().all()):
        raise AssertionError("Executed model metrics must be within [0, 1].")


def verify_files() -> None:
    for name in REQUIRED_TABLES:
        path = TABLES_DIR / name
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty table: {path}")

    for name in REQUIRED_SCREENSHOTS:
        path = SCREENSHOTS_DIR / name
        if not path.exists() or path.stat().st_size == 0:
            raise AssertionError(f"Missing or empty screenshot: {path}")
        with Image.open(path) as image:
            stat = ImageStat.Stat(image.convert("L"))
            if stat.var[0] == 0:
                raise AssertionError(f"Screenshot appears blank: {path}")


def verify_notebook() -> None:
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if [path.name for path in notebooks] != ["CDS6344_Full_Analysis_Notebook.ipynb"]:
        raise AssertionError(f"Expected exactly one notebook, found: {[path.name for path in notebooks]}")
    nb = nbformat.read(notebooks[0], as_version=4)
    nbformat.validate(nb)
    if len(nb.cells) < 20:
        raise AssertionError("Combined notebook is too small to cover the full workflow.")


def main() -> None:
    verify_audit()
    verify_class_distribution()
    verify_model_results()
    verify_files()
    verify_notebook()
    print("Verification passed: dataset audit, class counts, model tables, screenshots, and single notebook are valid.")


if __name__ == "__main__":
    main()
