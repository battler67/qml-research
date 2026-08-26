"""Disease-focused binary metrics with explicit positive-class semantics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else float("nan")


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    *,
    training_time_seconds: float,
    prediction_time_seconds: float,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    if not (len(y_true) == len(y_pred) == len(y_score)):
        raise ValueError("y_true, y_pred, and y_score lengths differ")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    try:
        auroc = float(roc_auc_score(y_true, y_score))
        auprc = float(average_precision_score(y_true, y_score))
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_score)
    except ValueError:
        auroc = auprc = float("nan")
        fpr = tpr = roc_thresholds = np.array([], dtype=float)
        pr_precision = pr_recall = pr_thresholds = np.array([], dtype=float)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity_score(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auroc": auroc,
        "auprc": auprc,
        "confusion_matrix": matrix.astype(int).tolist(),
        "training_time_seconds": float(training_time_seconds),
        "prediction_time_seconds": float(prediction_time_seconds),
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "precision_recall_curve": {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }
