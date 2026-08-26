"""Pre-registered, fixed classical baselines."""

from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def build_classical_model(name: str, seed: int) -> ClassifierMixin:
    if name == "logistic_regression":
        return LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=seed,
            solver="liblinear",
        )
    if name == "linear_svm":
        return SVC(C=1.0, kernel="linear", class_weight="balanced", random_state=seed)
    if name == "rbf_svm":
        return SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unsupported classical model: {name}")


def prediction_scores(model: ClassifierMixin, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X), dtype=float)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X), dtype=float)[:, 1]
    return np.asarray(model.predict(X), dtype=float)
