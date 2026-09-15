from __future__ import annotations

import json

import numpy as np
import pennylane as qml
import pytest
import torch

from qml_research.campaign import neural, storage
from qml_research.campaign.data import load_dataset, prepare, splits
from qml_research.campaign.kernels import embedding, fidelity, pauli_projection, states
from qml_research.ehr_ihd.models import HybridQMLP


@pytest.mark.parametrize("kind,repeats", [("angle", 1), ("iqp", 1), ("iqp", 2)])
def test_cached_fidelity_and_projection_match_circuit(kind, repeats, tmp_path):
    x = np.random.default_rng(14).normal(size=(4, 4))
    state = states(x, kind=kind, repeats=repeats, path=tmp_path, block_size=2)
    device = qml.device("default.qubit", wires=4)

    @qml.qnode(device)
    def overlap(left, right):
        embedding(left, 4, kind, repeats)
        qml.adjoint(embedding)(right, 4, kind, repeats)
        return qml.probs(wires=range(4))

    @qml.qnode(device)
    def projection(sample):
        embedding(sample, 4, kind, repeats)
        return tuple(
            qml.expval(op(wire)) for wire in range(4) for op in (qml.PauliX, qml.PauliY, qml.PauliZ)
        )

    kernel = fidelity(state, state)
    reference = np.array([[overlap(left, right)[0] for right in x] for left in x])
    assert np.allclose(kernel, reference, atol=1e-12)
    assert np.linalg.eigvalsh(kernel).min() >= -1e-10
    assert np.allclose(pauli_projection(state), [projection(row) for row in x], atol=1e-12)
    assert np.array_equal(states(x, kind=kind, repeats=repeats, path=tmp_path), state)
    if kind == "angle":
        analytic = np.prod(np.cos((x[:, None, :] - x[None, :, :]) / 2) ** 2, axis=-1)
        assert np.allclose(kernel, analytic, atol=1e-12)
    with pytest.raises(ValueError, match="identity"):
        states(x + 1, kind=kind, repeats=repeats, path=tmp_path)


def test_batched_hybrid_matches_sample_outputs_and_gradients():
    model = HybridQMLP(4, qubits=4, layers=2, seed=7)
    x = torch.randn(3, 4, dtype=torch.float64)
    latent = torch.tanh(model.projection(x)) * np.pi
    expected = torch.stack([torch.stack(model.qnode(row, model.quantum_weights)) for row in latent])
    actual = model.quantum_expectations(latent)
    assert torch.allclose(actual, expected, atol=1e-12)
    expected_grad = torch.autograd.grad(expected.sum(), model.quantum_weights, retain_graph=True)[0]
    actual_grad = torch.autograd.grad(actual.sum(), model.quantum_weights)[0]
    assert torch.allclose(actual_grad, expected_grad, atol=1e-12)


@pytest.mark.parametrize(
    "family", ["classical_mlp", "hqmlp", "reupload_vqc", "qcnn", "classical_cnn"]
)
def test_interrupted_batch_resume_matches_uninterrupted(family, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "OUTPUT", tmp_path / "stop-area")
    features = 784 if family == "classical_cnn" else 4
    x = np.random.default_rng(14).normal(size=(12, features))
    y = np.tile([0, 1], 6)
    spec = {"layers": 1, "lr": 0.01, "epochs": 3, "patience": 5, "batch_size": 3, "dropout": 0.1}
    whole = neural.make_model(family, features, spec, 7)
    expected = neural.train(
        whole, x[:8], y[:8], x[8:], y[8:], spec=spec, path=tmp_path / "whole", seed=7
    )
    interrupted = neural.make_model(family, features, spec, 7)

    def pause(step):
        if step == 2:
            raise storage.Paused("test interruption")

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
    restored = neural.make_model(family, features, spec, 99)
    actual = neural.train(
        restored, x[:8], y[:8], x[8:], y[8:], spec=spec, path=tmp_path / "resume", seed=7
    )
    assert actual["history"] == expected["history"]
    assert actual["steps"] == expected["steps"]
    for key, value in whole.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])
    progress = json.loads((tmp_path / "resume/progress.json").read_text())
    assert progress["completed"]


def test_training_resume_rejects_changed_data(tmp_path):
    x = np.zeros((8, 4))
    y = np.tile([0, 1], 4)
    spec = {"layers": 1, "lr": 0.01, "epochs": 1, "patience": 2, "batch_size": 4}
    model = neural.make_model("classical_mlp", 4, spec, 7)
    neural.train(model, x, y, x, y, spec=spec, path=tmp_path, seed=7)
    with pytest.raises(ValueError, match="identity"):
        neural.train(model, x + 1, y, x, y, spec=spec, path=tmp_path, seed=7)


def test_full_data_splits_and_training_only_preprocessing(tmp_path, monkeypatch):
    import qml_research.campaign.data as data_module

    monkeypatch.setattr(data_module, "OUTPUT", tmp_path)
    data = load_dataset("wdbc")
    seen = []
    for _, indices in splits(data, 42):
        a, b, c = indices
        assert len(a) + len(b) + len(c) == 569
        assert not set(data.groups[a]) & set(data.groups[b])
        assert not set(data.groups[a]) & set(data.groups[c])
        assert not set(data.groups[b]) & set(data.groups[c])
        seen.extend(c)
    assert sorted(seen) == list(range(569))
    _, indices = next(splits(data, 42))
    _, expected = prepare(data, indices)
    data.X.iloc[indices[2], 0] = 1e9
    _, actual = prepare(data, indices)
    assert np.array_equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1])


def test_kernel_resumes_after_saved_chunk(tmp_path, monkeypatch):
    import qml_research.campaign.kernels as kernels

    x = np.random.default_rng(9).normal(size=(6, 4))
    calls = 0

    def pause():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise storage.Paused("pause after a chunk")

    monkeypatch.setattr(kernels, "check_stop", pause)
    with pytest.raises(storage.Paused):
        states(x, kind="iqp", repeats=2, path=tmp_path, block_size=2)
    assert json.loads((tmp_path / "progress.json").read_text())["rows_completed"] == 2
    monkeypatch.setattr(kernels, "check_stop", lambda: None)
    resumed = states(x, kind="iqp", repeats=2, path=tmp_path, block_size=2)
    expected = states(x, kind="iqp", repeats=2)
    assert np.array_equal(resumed, expected)


def test_atomic_checkpoint_keeps_previous_file_on_failed_write(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    storage.atomic_torch(path, {"epoch": 3})

    def broken_save(value, stream):
        stream.write(b"incomplete")
        raise OSError("simulated interrupted write")

    monkeypatch.setattr(torch, "save", broken_save)
    with pytest.raises(OSError):
        storage.atomic_torch(path, {"epoch": 4})
    assert torch.load(path, weights_only=True) == {"epoch": 3}


def test_atomic_replace_retries_transient_file_sharing(tmp_path, monkeypatch):
    path = tmp_path / "progress.json"
    storage.atomic_json(path, {"epoch": 3})
    original = storage.os.replace
    attempts = []

    def transient(source, destination):
        attempts.append(1)
        if len(attempts) < 3:
            assert json.loads(path.read_text()) == {"epoch": 3}
            raise PermissionError("Windows file sharing")
        original(source, destination)

    monkeypatch.setattr(storage.os, "replace", transient)
    monkeypatch.setattr(storage.time, "sleep", lambda seconds: None)
    storage.atomic_json(path, {"epoch": 4})
    assert len(attempts) == 3
    assert json.loads(path.read_text()) == {"epoch": 4}


def test_completed_family_resume_does_not_refit(tmp_path, monkeypatch):
    import qml_research.campaign.runner as runner

    monkeypatch.setattr(runner, "OUTPUT", tmp_path)
    monkeypatch.setattr(
        runner, "trial_specs", lambda *args: [{"C": 1.0, "class_weight": "balanced"}]
    )
    data = load_dataset("wdbc")
    config = {"name": "fixture", "kernel_families": ["classical_logistic"]}
    fold, indices = next(splits(data, 42))
    first = runner.Fold(data, indices, 42, fold, config)
    expected = first.run_family("classical_logistic")
    second = runner.Fold(data, indices, 42, fold, config)

    def forbidden(*args):
        raise AssertionError("Completed family must not fit again")

    monkeypatch.setattr(second, "estimator", forbidden)
    assert second.run_family("classical_logistic") == expected
