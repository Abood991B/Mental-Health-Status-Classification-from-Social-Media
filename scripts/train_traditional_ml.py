from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_predictions, save_confusion_matrix
from src.feature_engineering import build_tfidf_vectorizer
from src.models.traditional_ml import build_traditional_models
from src.visualization import save_metric_bar


PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
SCREENSHOTS_DIR = RESULTS_DIR / "screenshots"
MODELS_DIR = ROOT / "models"


def load_split(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run scripts/run_eda_preprocessing.py first.")
    return pd.read_csv(path)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split("train")
    val_df = load_split("validation")
    test_df = load_split("test")
    labels = sorted(train_df["status"].unique())

    models = build_traditional_models()
    all_metrics = []
    all_reports = []
    sample_weight = compute_sample_weight(class_weight="balanced", y=train_df["status"])

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("tfidf", build_tfidf_vectorizer()),
                ("classifier", estimator),
            ]
        )

        fit_kwargs = {}
        if model_name == "Naive Bayes":
            fit_kwargs["classifier__sample_weight"] = sample_weight
        pipeline.fit(train_df["clean_text"], train_df["status"], **fit_kwargs)

        train_pred = pipeline.predict(train_df["clean_text"])
        val_pred = pipeline.predict(val_df["clean_text"])
        test_pred = pipeline.predict(test_df["clean_text"])

        test_metrics, report_df, cm_df = evaluate_predictions(test_df["status"], test_pred, labels, model_name)
        test_metrics["train_accuracy"] = float((train_pred == train_df["status"]).mean())
        test_metrics["validation_accuracy"] = float((val_pred == val_df["status"]).mean())
        test_metrics["test_accuracy"] = test_metrics["accuracy"]
        all_metrics.append(test_metrics)

        report_df.insert(0, "model", model_name)
        all_reports.append(report_df)
        safe_name = model_name.lower().replace(" ", "_")
        report_df.to_csv(TABLES_DIR / f"classification_report_{safe_name}.csv", index=False)
        cm_df.to_csv(TABLES_DIR / f"confusion_matrix_{safe_name}.csv")
        save_confusion_matrix(cm_df, f"Confusion Matrix: {model_name}", SCREENSHOTS_DIR / f"confusion_matrix_{safe_name}.png")
        joblib.dump(pipeline, MODELS_DIR / f"{safe_name}_pipeline.joblib")

    results_df = pd.DataFrame(all_metrics)
    ordered_columns = [
        "model",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
    ]
    results_df = results_df[ordered_columns]
    results_df.to_csv(TABLES_DIR / "traditional_ml_results.csv", index=False)
    pd.concat(all_reports, ignore_index=True).to_csv(TABLES_DIR / "traditional_ml_classification_reports.csv", index=False)
    save_metric_bar(results_df, "macro_f1", SCREENSHOTS_DIR / "traditional_ml_macro_f1_comparison.png", "Traditional ML Macro F1 Comparison")

    print(json.dumps(results_df.round(4).to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
