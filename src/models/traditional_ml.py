from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


def build_traditional_models() -> dict:
    return {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",
            n_jobs=None,
        ),
        "Linear SVM": LinearSVC(C=1.0, class_weight="balanced", random_state=42),
    }
