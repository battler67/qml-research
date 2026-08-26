from dataclasses import replace

import pytest

from qml_research.qcnn.config import (
    PROFILE_NAMES,
    load_qcnn_config,
    qcnn_config_from_checkpoint,
)


def test_all_qcnn_profiles_validate() -> None:
    for profile in PROFILE_NAMES:
        config = load_qcnn_config(profile)
        assert config.profile == profile
        assert config.reduced_features == config.qubits
        assert config.stages in {2, 3}


def test_cli_style_overrides_are_typed_and_fingerprinted() -> None:
    base = load_qcnn_config("smoke")
    changed = load_qcnn_config("smoke", ["train_size=20", "learning_rate=0.003", "seeds=[3,5]"])
    assert changed.train_size == 20
    assert changed.learning_rate == pytest.approx(0.003)
    assert changed.seeds == [3, 5]
    assert changed.fingerprint() != base.fingerprint()


def test_invalid_qcnn_shape_and_hardware_submission_are_rejected() -> None:
    config = load_qcnn_config("smoke")
    with pytest.raises(ValueError, match="reduced_features"):
        replace(config, reduced_features=8)
    with pytest.raises(ValueError, match="submission"):
        replace(config, real_hardware_submit=True)


def test_expensive_estimate_flags_full_and_eight_qubit_profiles() -> None:
    full = load_qcnn_config("full_dataset").estimated_resources()
    high = load_qcnn_config("high_memory").estimated_resources()
    assert full["estimated_quantum_forwards"] == (60 * (546 + 78) + 78 + 156) * 5
    assert full["expensive"]
    assert high["statevector_complex_amplitudes"] == 256
    assert "eight-qubit profile" in high["expensive_reasons"]


def test_smoke_estimate_counts_training_validation_and_test_circuits() -> None:
    estimate = load_qcnn_config("smoke").estimated_resources()
    assert estimate["estimated_quantum_forwards_per_fold_seed"] == 144
    assert estimate["estimated_quantum_forwards"] == 144


def test_legacy_checkpoint_config_has_a_narrow_migration() -> None:
    payload = load_qcnn_config("smoke").to_dict()
    payload["checkpoint_frequency"] = 1
    assert qcnn_config_from_checkpoint(payload).profile == "smoke"
    payload["unrecognized_scientific_setting"] = True
    with pytest.raises(ValueError, match="unknown configuration"):
        qcnn_config_from_checkpoint(payload)
