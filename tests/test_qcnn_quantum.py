import numpy as np
import pytest

torch = pytest.importorskip("torch")

from qml_research.qcnn.quantum import (  # noqa: E402
    HierarchicalQCNN,
    convolution_pairs,
    pooling_pairs,
)


@pytest.mark.qcnn
@pytest.mark.quantum
def test_qcnn_topology_parameter_counts_and_forward_shape() -> None:
    model = HierarchicalQCNN(4, seed=3)
    features = torch.zeros((3, 4), dtype=torch.float64)
    logits = model(features)
    assert logits.shape == (3,)
    assert model.trainable_quantum_parameters == 36
    assert model.trainable_total_parameters == 38
    assert pooling_pairs([0, 1, 2, 3]) == [(0, 1), (2, 3)]
    assert convolution_pairs([0, 1, 2, 3]) == [(0, 1), (2, 3), (1, 2), (3, 0)]


@pytest.mark.qcnn
@pytest.mark.quantum
def test_one_qcnn_optimization_step_has_finite_gradients_and_resources() -> None:
    model = HierarchicalQCNN(4, seed=5)
    features = torch.tensor([[-1.0, -0.5, 0.2, 0.8], [0.9, 0.4, -0.3, -0.7]], dtype=torch.float64)
    targets = torch.tensor([0.0, 1.0], dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features), targets)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    optimizer.step()
    resources = model.resource_summary()
    assert resources.qubits == 4
    assert resources.depth > 0
    assert resources.two_qubit_gates > 0
    assert resources.trainable_total_parameters == 38


def test_eight_qubit_parameter_count() -> None:
    model = HierarchicalQCNN(8, seed=1)
    assert model.trainable_quantum_parameters == 54
    assert model.trainable_total_parameters == 56
    assert np.log2(model.qubits) == model.stages
