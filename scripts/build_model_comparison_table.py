from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results" / "tables"


MODEL_REGISTRY = [
    {"model": "Naive Bayes", "family": "Traditional ML", "priority": "core", "source": "traditional_ml_results.csv"},
    {"model": "Logistic Regression", "family": "Traditional ML", "priority": "core", "source": "traditional_ml_results.csv"},
    {"model": "Linear SVM", "family": "Traditional ML", "priority": "core", "source": "traditional_ml_results.csv"},
    {"model": "CNN", "family": "Deep Learning", "priority": "recommended", "source": "deep_learning_results.csv"},
    {"model": "DistilBERT", "family": "Transformer", "priority": "recommended", "source": "transformer_results.csv"},
    {"model": "BiLSTM", "family": "Deep Learning", "priority": "optional", "source": "deep_learning_results.csv"},
    {"model": "GRU", "family": "Deep Learning", "priority": "optional", "source": "deep_learning_results.csv"},
    {"model": "BERT", "family": "Transformer", "priority": "optional", "source": "transformer_results.csv"},
    {"model": "RoBERTa", "family": "Transformer", "priority": "optional", "source": "transformer_results.csv"},
]

MODEL_ALIASES = {
    "bert-base-uncased": "BERT",
    "distilbert-base-uncased": "DistilBERT",
    "roberta-base": "RoBERTa",
}

METRIC_COLUMNS = [
    "train_accuracy",
    "validation_accuracy",
    "test_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
]


def load_results(filename: str) -> pd.DataFrame:
    path = TABLES_DIR / filename
    if not path.exists():
        return pd.DataFrame(columns=["model", *METRIC_COLUMNS])
    results = pd.read_csv(path)
    if "model" not in results.columns:
        raise ValueError(f"{filename} must contain a model column.")
    results["model"] = results["model"].replace(MODEL_ALIASES)
    for column in METRIC_COLUMNS:
        if column not in results.columns:
            results[column] = None
    return results[["model", *METRIC_COLUMNS]]


def main() -> None:
    traditional_path = TABLES_DIR / "traditional_ml_results.csv"
    if not traditional_path.exists():
        raise FileNotFoundError("Run scripts/train_traditional_ml.py before building the comparison table.")

    result_tables = {
        "traditional_ml_results.csv": load_results("traditional_ml_results.csv"),
        "deep_learning_results.csv": load_results("deep_learning_results.csv"),
        "transformer_results.csv": load_results("transformer_results.csv"),
    }

    rows = []
    for item in MODEL_REGISTRY:
        model = item["model"]
        matched = result_tables[item["source"]][result_tables[item["source"]]["model"] == model]
        if not matched.empty:
            row = matched.iloc[0].to_dict()
            row["status"] = "executed"
            row["notes"] = "Full test-set evaluation from local run"
        else:
            row = {
                "model": model,
                "train_accuracy": None,
                "validation_accuracy": None,
                "test_accuracy": None,
                "macro_f1": None,
                "weighted_f1": None,
                "macro_precision": None,
                "macro_recall": None,
                "status": "pending",
                "notes": "Selected for main run"
                if item["priority"] == "recommended"
                else "Optional extension not executed in final laptop run",
            }
        row["family"] = item["family"]
        row["priority"] = item["priority"]
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison = comparison[["model", "family", "priority", *METRIC_COLUMNS, "status", "notes"]]
    comparison.to_csv(TABLES_DIR / "model_comparison_table.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
