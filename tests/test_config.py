from pathlib import Path

import pytest

from qml_research.config import ExperimentConfig, load_config


def test_smoke_config_loads_and_hash_is_stable() -> None:
    config = load_config(Path("configs/smoke.yaml"))
    assert config.dataset == "iris"
    assert config.backend == "default.qubit"
    assert config.optimizer == "adam"
    assert config.learning_rate == 0.05
    assert config.digest({"fold": 0}) == config.digest({"fold": 0})
    assert config.digest({"fold": 0}) != config.digest({"fold": 1})


def test_invalid_scientific_options_are_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported dataset"):
        ExperimentConfig(name="bad", dataset="private_clinical")
    with pytest.raises(ValueError, match="feature counts"):
        ExperimentConfig(name="bad", dataset="iris", feature_counts=(30,))
    with pytest.raises(ValueError, match="optimizer"):
        ExperimentConfig(name="bad", dataset="iris", optimizer="cobyla")
    with pytest.raises(ValueError, match="default.qubit"):
        ExperimentConfig(name="bad", dataset="iris", backend="hardware")
