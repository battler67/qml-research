import json

import numpy as np

from qml_research.results import ResultStore, load_records, records_to_frame


def test_result_serialization_and_resume(tmp_path) -> None:
    store = ResultStore(tmp_path)
    path = store.write(
        "abc",
        {
            "status": "completed",
            "experiment": "unit",
            "run": {"dataset": "iris", "model": "qksvm"},
            "metrics": {"balanced_accuracy": np.float64(0.9), "auroc": float("nan")},
            "quantum_resources": {"circuit_executions": np.int64(3)},
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metrics"]["auroc"] is None
    assert payload["quantum_resources"]["circuit_executions"] == 3
    assert store.exists("abc")
    frame = records_to_frame(load_records(tmp_path))
    assert frame.loc[0, "balanced_accuracy"] == 0.9


def test_kernel_matrix_serialization(tmp_path) -> None:
    store = ResultStore(tmp_path)
    matrix = np.eye(2)
    path = store.write_kernel_matrix("kernel", matrix, matrix[:1])
    loaded = np.load(path)
    assert np.array_equal(loaded["train"], matrix)
    assert loaded["test"].shape == (1, 2)
