import pytest

from qml_research.config import load_config
from qml_research.results import ResultStore
from qml_research.runner import _project_full_qml_runtime, run_experiments


@pytest.mark.quantum
def test_smoke_runner_completes(tmp_path) -> None:
    config = load_config("configs/smoke.yaml")
    config = type(config)(**{**config.to_dict(), "output_dir": str(tmp_path)})
    summary = run_experiments(config)
    assert summary["completed"] > 0
    assert summary["failed"] == 0


def test_full_qksvm_runtime_projection_is_quadratic(tmp_path) -> None:
    config = load_config("configs/wdbc_benchmark.yaml")
    run = {
        "dataset": "wdbc",
        "sample_size": "full",
        "seed": 42,
        "model": "qksvm",
        "reduction": "pca",
        "feature_count": 4,
        "feature_map": "angle",
        "vqc_layers": None,
        "quantum_condition": "ideal",
    }
    previous = {**run, "sample_size": 200, "fold": 0}
    store = ResultStore(tmp_path)
    store.write(
        "timing",
        {
            "status": "completed",
            "experiment": config.name,
            "run": previous,
            "metrics": {"training_time_seconds": 10.0, "prediction_time_seconds": 2.0},
        },
    )

    projection = _project_full_qml_runtime(store, config, run, full_sample_count=569)

    assert projection == pytest.approx(12.0 * (569 / 200) ** 2)
