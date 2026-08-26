import numpy as np
import pytest

from qml_research.quantum import VariationalQuantumClassifier


@pytest.mark.quantum
def test_vqc_prediction_shape_history_and_resources() -> None:
    X = np.array([[-1.0, -1.0], [-0.5, -0.2], [0.5, 0.2], [1.0, 1.0]])
    y = np.array([0, 0, 1, 1])
    model = VariationalQuantumClassifier(2, layers=1, steps=2, batch_size=4, seed=42)
    model.fit(X, y)
    scores = model.decision_function(X)
    predictions = (scores >= 0).astype(int)
    assert scores.shape == (4,)
    assert predictions.shape == (4,)
    assert len(model.loss_history_) == 2
    assert model.fit_summary_.trainable_parameters == 7
    assert model.execution_count > 0
    resources = model.resources(X[0])
    assert resources["num_qubits"] == 2
    assert resources["circuit_depth"] is not None
    assert resources["total_gate_count"] > 0
