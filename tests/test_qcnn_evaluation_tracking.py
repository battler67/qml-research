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
