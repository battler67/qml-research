from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from qml_research.ehr_ihd.models import HybridQMLP, PauliStatevectorKernel


@pytest.mark.quantum
@pytest.mark.parametrize("features", [4, 6, 8, 10])
def test_pauli_kernel_shapes_symmetry_and_cache_roundtrip(features: int, tmp_path: Path) -> None:
    rng = np.random.default_rng(features)
    values = rng.uniform(0, np.pi, size=(2, features))
    kernel = PauliStatevectorKernel(features, reps=1)
    matrix = kernel.matrix(values, values, symmetric=True)
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    cache = tmp_path / f"kernel-{features}.npz"
    np.savez_compressed(cache, train=matrix)
    assert np.allclose(np.load(cache)["train"], matrix)


@pytest.mark.quantum
def test_qmlp_forward_backward_and_save_load(tmp_path: Path) -> None:
    model = HybridQMLP(4, qubits=4, layers=1, seed=42)
    values = torch.zeros((3, 4), dtype=torch.float64)
    target = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    loss = torch.nn.BCEWithLogitsLoss()(model(values), target)
    loss.backward()
    assert model.quantum_weights.grad is not None
    path = tmp_path / "qmlp.pt"
    torch.save(model.state_dict(), path)
    restored = HybridQMLP(4, qubits=4, layers=1, seed=9)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert torch.allclose(model(values), restored(values))
