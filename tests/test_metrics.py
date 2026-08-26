import numpy as np
import pytest

from qml_research.evaluation import classification_metrics, specificity_score
from qml_research.reports import METRIC_COLUMNS, _scalar_metrics


def test_sensitivity_specificity_and_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    y_score = np.array([0.1, 0.7, 0.8, 0.2])
    assert specificity_score(y_true, y_pred) == pytest.approx(0.5)
    metrics = classification_metrics(
        y_true,
        y_pred,
        y_score,
        training_time_seconds=1.0,
        prediction_time_seconds=0.1,
    )
    assert metrics["sensitivity"] == pytest.approx(0.5)
    assert metrics["specificity"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["auroc"] == pytest.approx(0.75)
    bootstrap_metrics = _scalar_metrics(y_true, y_pred, y_score)
    for metric in METRIC_COLUMNS:
        assert bootstrap_metrics[metric] == pytest.approx(metrics[metric])
