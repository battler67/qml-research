import numpy as np
import pytest

from qml_research.qcnn.evaluation import (
    choose_threshold,
    evaluate_scores,
    stratified_bootstrap_interval,
)
from qml_research.qcnn.tracking import RunDirectory, stable_hash


def test_threshold_metrics_and_bootstrap_include_disease_errors(tmp_path) -> None:
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.3, 0.7, 0.4, 0.8, 0.9])
    threshold = choose_threshold(y, scores, "youden_j")
    metrics = evaluate_scores(
        y, scores, threshold=threshold, training_time_seconds=1.0, prediction_time_seconds=0.1
    )
    interval = stratified_bootstrap_interval(
        y,
        scores,
        metric="balanced_accuracy",
        threshold=threshold,
        iterations=50,
        seed=2,
    )
    assert metrics["false_negatives"] >= 0
    assert metrics["false_positives"] >= 0
    assert interval["lower"] <= interval["median"] <= interval["upper"]
    run = RunDirectory(tmp_path, {"b": 2, "a": 1})
    run.write("record.json", {"status": "completed"})
    assert run.completed()
    assert run.run_id == stable_hash({"a": 1, "b": 2})


def test_fixed_threshold() -> None:
    assert choose_threshold(np.array([0, 1]), np.array([0.1, 0.9]), "fixed_0.5") == 0.5
    with pytest.raises(ValueError):
        choose_threshold(np.array([0, 1]), np.array([0.1, 0.9]), "unknown")


@pytest.mark.parametrize(
    "metric", ["auroc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
)
def test_bootstrap_matches_full_metric_reference(metric: str) -> None:
    # Unequal class counts, tied scores, and scores exactly at the threshold.
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.5, 0.5, 0.9, 0.2, 0.5, 0.8])
    rng = np.random.default_rng(23)
    values = []
    for _ in range(20):
        sampled = np.concatenate(
            [
                rng.choice(np.flatnonzero(y == label), np.sum(y == label), replace=True)
                for label in (0, 1)
            ]
        )
        rng.shuffle(sampled)
        values.append(
            evaluate_scores(
                y[sampled],
                scores[sampled],
                threshold=0.5,
                training_time_seconds=0.0,
                prediction_time_seconds=0.0,
            )[metric]
        )
    reference = dict(
        zip(("lower", "median", "upper"), np.percentile(values, [2.5, 50, 97.5]), strict=True)
    )
    actual = stratified_bootstrap_interval(
        y,
        scores,
        metric=metric,
        threshold=0.5,
        iterations=20,
        seed=23,
    )
    assert actual == pytest.approx(reference, abs=1e-14)
