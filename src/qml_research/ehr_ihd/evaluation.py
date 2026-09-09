"""Validation-only calibration, clinical metrics, uncertainty, and explanations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def fit_platt_calibrator(scores: np.ndarray, target: np.ndarray) -> LogisticRegression:
    calibrator = LogisticRegression(random_state=42)
    calibrator.fit(np.asarray(scores, dtype=float).reshape(-1, 1), target)
    return calibrator


def calibrated_probabilities(calibrator: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(np.asarray(scores, dtype=float).reshape(-1, 1))[:, 1]


def select_threshold(target: np.ndarray, probabilities: np.ndarray, strategy: str) -> float:
    target = np.asarray(target, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if strategy == "youden_j":
        fpr, tpr, thresholds = roc_curve(target, probabilities)
        finite = np.isfinite(thresholds)
        return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    if len(thresholds) == 0:
        return 0.5
    if strategy in {"max_f1", "max_f2"}:
        beta = 1.0 if strategy == "max_f1" else 2.0
        denom = beta**2 * precision[:-1] + recall[:-1]
        score = np.divide(
            (1 + beta**2) * precision[:-1] * recall[:-1],
            denom,
            out=np.zeros_like(denom),
            where=denom > 0,
        )
        return float(thresholds[np.argmax(score)])
    if strategy == "sensitivity_at_least_0_80":
        candidates = np.where(recall[:-1] >= 0.80)[0]
        return float(thresholds[candidates[-1]]) if len(candidates) else 0.5
    raise ValueError(f"unsupported threshold strategy: {strategy}")


def expected_calibration_error(
    target: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(target)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (probability >= lower) & (probability < upper if upper < 1 else probability <= upper)
        if mask.any():
            observed_gap = abs(float(target[mask].mean()) - float(probability[mask].mean()))
            result += mask.mean() * observed_gap
    return float(result) if total else float("nan")


def binary_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float,
    *,
    training_time_seconds: float,
    inference_time_seconds: float,
    peak_memory_mb: float,
) -> dict[str, Any]:
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    return {
        "patient_count": int(len(target)),
        "positive_count": int(target.sum()),
        "prevalence": float(target.mean()),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(target, prediction)),
        "auroc": float(roc_auc_score(target, probability)),
        "auprc": float(average_precision_score(target, probability)),
        "precision_ppv": float(precision_score(target, prediction, zero_division=0)),
        "recall_sensitivity": float(recall_score(target, prediction, zero_division=0)),
        "specificity": float(specificity),
        "npv": float(npv),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "f2": float(fbeta_score(target, prediction, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(target, prediction)),
        "brier_score": float(brier_score_loss(target, probability)),
        "expected_calibration_error": expected_calibration_error(target, probability),
        "selected_threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "training_time_seconds": float(training_time_seconds),
        "inference_time_seconds": float(inference_time_seconds),
        "peak_memory_mb": float(peak_memory_mb),
    }


def bootstrap_intervals(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {"auroc": [], "auprc": [], "brier_score": []}
    for _ in range(iterations):
        sample = rng.integers(0, len(target), size=len(target))
        sampled_target = target[sample]
        if len(np.unique(sampled_target)) < 2:
            continue
        sampled_probability = probability[sample]
        values["auroc"].append(float(roc_auc_score(sampled_target, sampled_probability)))
        values["auprc"].append(float(average_precision_score(sampled_target, sampled_probability)))
        values["brier_score"].append(float(brier_score_loss(sampled_target, sampled_probability)))
    return {
        name: {
            "lower_95": float(np.percentile(metric_values, 2.5)),
            "upper_95": float(np.percentile(metric_values, 97.5)),
            "valid_resamples": len(metric_values),
        }
        for name, metric_values in values.items()
        if metric_values
    }


def permutation_explanation(
    estimator: Any,
    values: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    *,
    seed: int,
) -> dict[str, Any]:
    result = permutation_importance(
        estimator,
        values,
        target,
        scoring="average_precision",
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    return {
        "scope": "model_behaviour_not_clinical_causality",
        "metric": "average_precision",
        "features": [
            {
                "feature": name,
                "importance_mean": float(result.importances_mean[index]),
                "importance_std": float(result.importances_std[index]),
            }
            for index, name in sorted(
                enumerate(feature_names), key=lambda item: -result.importances_mean[item[0]]
            )
        ],
    }


def perturbation_explanation(
    score_function: Callable[[np.ndarray], np.ndarray],
    values: np.ndarray,
    feature_names: list[str],
) -> dict[str, Any]:
    baseline = np.asarray(score_function(values), dtype=float)
    features = []
    for index, name in enumerate(feature_names):
        perturbed = values.copy()
        perturbed[:, index] = 0.0
        delta = np.asarray(score_function(perturbed), dtype=float) - baseline
        features.append({"feature": name, "mean_score_change": float(delta.mean())})
    return {
        "scope": "model_behaviour_not_clinical_causality",
        "method": "zero_reference_feature_ablation",
        "features": sorted(features, key=lambda item: -abs(item["mean_score_change"])),
    }
