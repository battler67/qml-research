from pathlib import Path

import numpy as np

from qml_research.evaluation import classification_metrics
from qml_research.reports import generate_reports
from qml_research.results import ResultStore
from qml_research.visualization import generate_all_plots


def _record(model: str, predictions: np.ndarray) -> dict:
    y_true = np.array([0, 0, 1, 1])
    scores = predictions.astype(float)
    metrics = classification_metrics(
        y_true,
        predictions,
        scores,
        training_time_seconds=0.1,
        prediction_time_seconds=0.01,
    )
    is_quantum = model == "qksvm"
    return {
        "status": "completed",
        "experiment": "report_smoke",
        "run": {
            "dataset": "wdbc",
            "sample_size": 20,
            "seed": 42,
            "fold": 0,
            "model": model,
            "reduction": "pca",
            "feature_count": 2,
            "feature_map": "angle" if is_quantum else None,
            "vqc_layers": None,
            "quantum_condition": "ideal" if is_quantum else None,
            "preprocessing": "fold_local_quantum_matched",
        },
        "metrics": metrics,
        "predictions": {
            "y_true": y_true.tolist(),
            "y_pred": predictions.tolist(),
            "y_score": scores.tolist(),
        },
        "fit": {"converged": True},
        "quantum_resources": (
            {
                "backend": "default.qubit",
                "shots": None,
                "noise_strength": 0.0,
                "num_qubits": 2,
                "circuit_depth": 2,
                "total_gate_count": 4,
                "two_qubit_gate_count": 0,
                "trainable_parameters": 0,
                "circuit_executions": 10,
            }
            if is_quantum
            else {}
        ),
        "kernel_quality": (
            {
                "symmetric_error": 0.0,
                "diagonal_max_error": 0.0,
                "minimum_eigenvalue": 0.1,
                "negative_eigenvalues": 0,
            }
            if is_quantum
            else {}
        ),
        "artifacts": {"circuit_text": "0: --RY--"} if is_quantum else {},
    }


def test_report_and_all_plots_regenerate_from_serialized_records(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    store = ResultStore(raw)
    store.write("quantum", _record("qksvm", np.array([0, 0, 1, 1])))
    store.write("classical", _record("logistic_regression", np.array([0, 1, 0, 1])))

    outputs = generate_reports(
        raw,
        tmp_path / "tables",
        tmp_path / "research",
        bootstrap_repetitions=20,
    )
    figures = generate_all_plots(raw, tmp_path / "figures")

    assert all(path.is_file() for path in outputs.values())
    assert len(figures) == 11
    assert all(path.is_file() and path.stat().st_size > 0 for path in figures)
