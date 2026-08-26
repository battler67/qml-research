"""Threshold selection, binary metrics, and uncertainty estimates."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_curve

from qml_research.evaluation.metrics import classification_metrics


def choose_threshold(y_true: np.ndarray, scores: np.ndarray, strategy: str) -> float:
    if strategy == "fixed_0.5":
        return 0.5
    if strategy != "youden_j":
        raise ValueError(f"unsupported threshold strategy: {strategy}")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_true, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    candidate_indices = np.flatnonzero(finite)
    best_local = int(np.argmax((true_positive_rate - false_positive_rate)[finite]))
    return float(thresholds[candidate_indices[best_local]])


def evaluate_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    training_time_seconds: float,
    prediction_time_seconds: float,
) -> dict[str, Any]:
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    metrics = classification_metrics(
        np.asarray(y_true, dtype=int),
        predictions,
        scores,
        training_time_seconds=training_time_seconds,
        prediction_time_seconds=prediction_time_seconds,
    )
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    metrics.update(
        {
            "threshold": float(threshold),
            "false_positives": int(matrix[0, 1]),
            "false_negatives": int(matrix[1, 0]),
        }
    )
    return metrics


def stratified_bootstrap_interval(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    metric: str,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    by_class = {label: np.flatnonzero(y_true == label) for label in (0, 1)}
    values: list[float] = []
    for _ in range(iterations):
        sampled = np.concatenate(
            [rng.choice(indices, len(indices), replace=True) for indices in by_class.values()]
        )
        rng.shuffle(sampled)
        result = evaluate_scores(
            y_true[sampled],
            scores[sampled],
            threshold=threshold,
            training_time_seconds=0.0,
            prediction_time_seconds=0.0,
        )
        value = result.get(metric)
        if value is not None and np.isfinite(value):
            values.append(float(value))
    if not values:
        return {"lower": float("nan"), "median": float("nan"), "upper": float("nan")}
    lower, median, upper = np.percentile(values, [2.5, 50.0, 97.5])
    return {"lower": float(lower), "median": float(median), "upper": float(upper)}
