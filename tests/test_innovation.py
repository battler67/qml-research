from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from qml_research.campaign import neural, storage
from qml_research.campaign.data import Dataset
from qml_research.innovation import experiment
from qml_research.innovation.data import partition, preprocess, seal
from qml_research.innovation.models import GatedResidual
from qml_research.innovation.report import intervals, measures, rate_intervals


def fixture_data():
    rng = np.random.default_rng(7)
    x = pd.DataFrame(rng.normal(size=(100, 8)), columns=list("abcdefgh"))
    x.iloc[0, 0] = np.nan
    return Dataset("fixture", x, np.tile([0, 1], 50), [], {"sha256": "fixture"})


def test_locked_test_and_train_only_preprocessing():
    data = fixture_data()
    train, val, test = partition(data, 42, {})
    assert np.array_equal(test, partition(data, 123, {})[2])
    assert not set(train) & set(test)
    assert not set(val) & set(test)
    assert len(set(train) | set(val) | set(test)) == 100
    _, originals, reduced = preprocess(data, train, val, 4)
    data.X.iloc[test, :] = 1e12
    _, originals_changed, reduced_changed = preprocess(data, train, val, 4)
    for before, after in zip(originals + reduced, originals_changed + reduced_changed, strict=True):
        assert np.array_equal(before, after)


@pytest.mark.parametrize("variant", experiment.VARIANTS)
def test_residual_resume_and_gradient(variant, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "OUTPUT", tmp_path / "stop")
    rng = np.random.default_rng(4)
    x = rng.normal(size=(12, 5))
    y = np.tile([0, 1], 6)
    spec = {"layers": 2, "lr": 0.01, "epochs": 2, "patience": 3, "batch_size": 4}
    model = GatedResidual(4, spec, 7, variant)
    model(torch.tensor(x)).sum().backward()
    if variant in {"quantum", "ungated", "separable"}:
        assert torch.isfinite(model.angles.grad).all()
        assert model.angles.grad.abs().sum() > 0
    whole = neural.train(
        model, x[:8], y[:8], x[8:], y[8:], spec=spec, path=tmp_path / "whole", seed=7
    )
    interrupted = GatedResidual(4, spec, 7, variant)

    def pause(step):
        if step == 1:
            raise storage.Paused("test")

    with pytest.raises(storage.Paused):
        neural.train(
            interrupted,
            x[:8],
            y[:8],
            x[8:],
            y[8:],
            spec=spec,
            path=tmp_path / "resume",
            seed=7,
            on_batch=pause,
        )
    restored = GatedResidual(4, spec, 99, variant)
    resumed = neural.train(
        restored, x[:8], y[:8], x[8:], y[8:], spec=spec, path=tmp_path / "resume", seed=7
    )
    assert whole["history"] == resumed["history"]
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_no_quantum_is_exact_calibration_and_gate_suppresses_residual():
    spec = {"layers": 1}
    x = torch.randn(3, 5, dtype=torch.float64)
    model = GatedResidual(4, spec, 7, "no_quantum")
    assert torch.equal(model(x), x[:, 0])
    gated = GatedResidual(4, spec, 7, "quantum")
    x[:, 0] = 100
    assert torch.equal(gated(x), x[:, 0])


def test_metric_ties_and_paired_bootstrap():
    y = np.array([0, 1, 0, 1, 0, 1])
    score = np.array([0.2, 0.4, 0.4, 0.8, 0.1, 0.4])
    pred = (score >= 0.4).astype(int)
    actual = measures(y, score, pred)
    assert actual[0] == pytest.approx(roc_auc_score(y, score))
    assert actual[1] == pytest.approx(average_precision_score(y, score))
    _, sampled = intervals(y, np.tile(score, (2, 3, 1)), np.tile(pred, (2, 3, 1)), 20)
    assert np.array_equal(sampled[:, 0], sampled[:, 1])


def test_rate_intervals_do_not_claim_certainty_at_perfect_scores():
    y = np.tile([0, 1], 30)
    low, high = rate_intervals(y, np.tile(y, (3, 1)))
    # Exact all-success lower limit with simultaneous alpha/3 per seed.
    assert np.allclose(low, (0.05 / 6) ** (1 / 30))
    assert np.array_equal(high, [1.0, 1.0])
    assert np.all(low < 1)


def test_rate_intervals_match_exact_single_seed():
    from scipy.stats import binomtest

    y = np.array([0] * 10 + [1] * 20)
    prediction = np.array([0] * 8 + [1] * 2 + [1] * 15 + [0] * 5)
    bounds = rate_intervals(y, [prediction])
    for j, (k, n) in enumerate([(15, 20), (8, 10)]):
        expected = binomtest(k, n).proportion_ci()
        assert np.allclose(bounds[:, j], [expected.low, expected.high])


def test_budget_parity_and_sealed_configuration(tmp_path):
    config = {"trials_per_family": 3, "epochs": 2, "patience": 2, "batch_size": 4}
    for family in experiment.FAMILIES:
        assert len(experiment.specifications(family, config)) == 3
    path = tmp_path / "config.json"
    seal(path, config)
    with pytest.raises(ValueError, match="Immutable"):
        seal(path, {**config, "epochs": 3})


def test_smoke_forbids_test_and_incomplete_selection_blocks_evaluation(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Smoke"):
        experiment.evaluate({"allow_test": False})
    monkeypatch.setattr(experiment, "initialize", lambda config: (fixture_data(), tmp_path))
    with pytest.raises(ValueError, match="Incomplete"):
        experiment.evaluate({"allow_test": True, "seeds": [42]})
    assert not (tmp_path / "holdout_unsealed.json").exists()


def test_validation_selection_resume_skips_completed_trials(tmp_path, monkeypatch):
    data = fixture_data()
    monkeypatch.setattr(experiment, "initialize", lambda config: (data, tmp_path))
    monkeypatch.setattr(experiment, "OUTPUT", tmp_path)
    monkeypatch.setattr(experiment, "FAMILIES", ["logistic_original"])
    config = {
        "name": "fixture",
        "seeds": [42],
        "qubits": 4,
        "trials_per_family": 2,
        "allow_test": False,
    }
    experiment.select(config)
    path = tmp_path / "seed-42/logistic_original/selected.json"
    original = json.loads(path.read_text())

    def forbidden(*args):
        raise AssertionError("Refitting completed trials")

    monkeypatch.setattr(experiment, "estimator", forbidden)
    experiment.select(config)
    assert original == json.loads(path.read_text())
