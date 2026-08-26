import numpy as np
import pytest

from qml_research.quantum import QuantumKernel


@pytest.mark.quantum
@pytest.mark.parametrize("feature_map", ["angle", "iqp"])
def test_kernel_self_similarity_symmetry_dimensions_and_psd(feature_map: str) -> None:
    X = np.array([[0.1, -0.2], [0.3, 0.5], [-0.7, 0.2]])
    kernel = QuantumKernel(2, feature_map=feature_map, seed=42)
    matrix = kernel.matrix(X, X, symmetric=True)
    quality = kernel.quality(matrix)
    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-8)
    assert np.allclose(matrix, matrix.T, atol=1e-10)
    assert quality.minimum_eigenvalue >= -1e-8
    assert kernel.execution_count == 6
    assert kernel.matrix(X[:1], X).shape == (1, 3)
    resources = kernel.resources(X[0])
    assert resources["num_qubits"] == 2
    assert resources["circuit_depth"] is not None
    assert resources["total_gate_count"] > 0


@pytest.mark.quantum
def test_finite_shot_kernel_reports_numerical_quality() -> None:
    X = np.array([[0.1, -0.2], [0.3, 0.5]])
    kernel = QuantumKernel(2, shots=100, seed=7)
    matrix = kernel.matrix(X, X, symmetric=True)
    quality = kernel.quality(matrix)
    assert matrix.shape == (2, 2)
    assert isinstance(quality.negative_eigenvalues, int)
