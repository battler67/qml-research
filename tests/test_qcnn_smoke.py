import json
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")

from qml_research.qcnn.config import load_qcnn_config  # noqa: E402
from qml_research.qcnn.data import DatasetSplits, Split  # noqa: E402
from qml_research.qcnn.models import SmallCNN, prepare_image_tensor  # noqa: E402
from qml_research.qcnn.runner import (  # noqa: E402
    export_hardware_bundle,
    run_experiment,
    verify_research,
)
from qml_research.qcnn.training import normalize_binary_logits  # noqa: E402


def _synthetic_split(count: int, seed: int, offset: int) -> Split:
    rng = np.random.default_rng(seed)
    labels = np.tile([0, 1], count // 2)
    images = np.empty((count, 28, 28), dtype=np.uint8)
    for index, label in enumerate(labels):
        center = 55 if label == 0 else 200
        images[index] = np.clip(rng.normal(center, 12, size=(28, 28)), 0, 255)
    return Split(images, labels, np.arange(offset, offset + count))


@pytest.mark.qcnn
@pytest.mark.quantum
def test_synthetic_smoke_profile_completes_and_writes_artifacts(tmp_path, monkeypatch) -> None:
    splits = DatasetSplits(
        name="breastmnist",
        train=_synthetic_split(8, 1, 0),
        validation=_synthetic_split(4, 2, 100),
        test=_synthetic_split(4, 3, 200),
        provenance={"source": "synthetic unit fixture", "md5": "fixture"},
        is_image=True,
    )
    splits.validate()
    monkeypatch.setattr("qml_research.qcnn.runner.load_splits", lambda *args, **kwargs: splits)
    config = replace(
        load_qcnn_config("smoke"),
        output_dir=str(tmp_path),
        train_size=8,
        validation_size=4,
        test_size=4,
        models=["qcnn", "logistic", "small_cnn"],
        epochs=1,
        image_epochs=1,
        early_stopping_patience=1,
        bootstrap_iterations=20,
        seeds=[3],
    )
    result = run_experiment(config)
    assert result["completed"] == 3
    assert result["failed"] == 0
    assert len(list((tmp_path / "runs").glob("*/record.json"))) == 3
    assert len(list((tmp_path / "runs").glob("*/predictions.csv"))) == 3
    assert len(list((tmp_path / "runs").glob("*/checkpoint.pt"))) == 2
    assert (tmp_path / "reports" / "normalized_results.csv").exists()


def test_small_cnn_forward_and_research_gate() -> None:
    model = SmallCNN()
    tensor = prepare_image_tensor(np.zeros((2, 28, 28), dtype=np.uint8), "native_28x28")
    assert model(tensor).shape == (2,)
    assert verify_research()["status"] == "verified"


def test_binary_logit_normalization_accepts_native_and_torchvision_heads() -> None:
    import torch

    assert normalize_binary_logits(torch.zeros(8)).shape == (8,)
    assert normalize_binary_logits(torch.zeros(8, 1)).shape == (8,)
    assert normalize_binary_logits(torch.zeros(1, 1)).shape == (1,)
    with pytest.raises(ValueError, match="binary model"):
        normalize_binary_logits(torch.zeros(2, 2))


@pytest.mark.qcnn
@pytest.mark.quantum
def test_hardware_profile_exports_without_submission(tmp_path) -> None:
    config = replace(
        load_qcnn_config("real_hardware_ready"),
        output_dir=str(tmp_path),
    )
    result = export_hardware_bundle(config)
    assert result["status"] == "ready_not_submitted"
    assert result["resources"]["qubits"] == 4
    assert (tmp_path / "hardware_export" / "circuit.txt").exists()


def test_model_failure_is_serialized_and_other_results_can_report(tmp_path, monkeypatch) -> None:
    splits = DatasetSplits(
        name="breastmnist",
        train=_synthetic_split(8, 1, 0),
        validation=_synthetic_split(4, 2, 100),
        test=_synthetic_split(4, 3, 200),
        provenance={"source": "synthetic unit fixture", "md5": "fixture"},
        is_image=True,
    )
    monkeypatch.setattr("qml_research.qcnn.runner.load_splits", lambda *args, **kwargs: splits)

    def fail_training(*args, **kwargs):
        raise RuntimeError("intentional training failure")

    monkeypatch.setattr("qml_research.qcnn.runner.train_image_model", fail_training)
    config = replace(
        load_qcnn_config("smoke"),
        output_dir=str(tmp_path),
        models=["small_cnn"],
        seeds=[5],
    )
    result = run_experiment(config)
    assert result["failed"] == 1
    record_path = next((tmp_path / "runs").glob("*/record.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["error"]["type"] == "RuntimeError"
