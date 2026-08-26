"""Genuine hierarchical QCNN implemented with PennyLane and Torch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pennylane as qml
import torch
from torch import nn


def convolution_block(weights: torch.Tensor, wires: tuple[int, int]) -> None:
    """Fifteen-parameter SU(4)-style local convolution block."""

    qml.U3(weights[0], weights[1], weights[2], wires=wires[0])
    qml.U3(weights[3], weights[4], weights[5], wires=wires[1])
    qml.IsingXX(weights[6], wires=wires)
    qml.IsingYY(weights[7], wires=wires)
    qml.IsingZZ(weights[8], wires=wires)
    qml.U3(weights[9], weights[10], weights[11], wires=wires[0])
    qml.U3(weights[12], weights[13], weights[14], wires=wires[1])


def pooling_block(weights: torch.Tensor, source: int, sink: int) -> None:
    """Unitary source-to-sink pooling; source is retired by the caller."""

    qml.CRZ(weights[0], wires=(source, sink))
    qml.PauliX(wires=source)
    qml.CRX(weights[1], wires=(source, sink))
    qml.PauliX(wires=source)
    qml.CRY(weights[2], wires=(source, sink))


def convolution_pairs(active: list[int]) -> list[tuple[int, int]]:
    if len(active) < 2 or len(active) % 2:
        raise ValueError("active wire count must be an even power-of-two stage")
    pairs = [(active[index], active[index + 1]) for index in range(0, len(active), 2)]
    if len(active) > 2:
        pairs.extend(
            (active[index], active[(index + 1) % len(active)]) for index in range(1, len(active), 2)
        )
    return pairs


def pooling_pairs(active: list[int]) -> list[tuple[int, int]]:
    return [(active[index], active[index + 1]) for index in range(0, len(active), 2)]


@dataclass(frozen=True)
class QCNNResources:
    qubits: int
    depth: int
    total_gates: int
    two_qubit_gates: int
    gate_types: dict[str, int]
    trainable_quantum_parameters: int
    trainable_total_parameters: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "qubits": self.qubits,
            "depth": self.depth,
            "total_gates": self.total_gates,
            "two_qubit_gates": self.two_qubit_gates,
            "gate_types": self.gate_types,
            "trainable_quantum_parameters": self.trainable_quantum_parameters,
            "trainable_total_parameters": self.trainable_total_parameters,
        }


class HierarchicalQCNN(nn.Module):
    """Four- or eight-qubit inverse-MERA-style binary classifier."""

    def __init__(
        self,
        qubits: int = 4,
        *,
        backend: str = "lightning.qubit",
        shots: int | None = None,
        gradient_method: str = "adjoint",
        init_scale: float = 0.05,
        noise_probability: float = 0.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if qubits not in {4, 8}:
            raise ValueError("HierarchicalQCNN supports four or eight qubits")
        if noise_probability and backend != "default.mixed":
            raise ValueError("noise_probability requires default.mixed")
        self.qubits = qubits
        self.stages = int(np.log2(qubits))
        self.backend = backend
        self.shots = shots
        self.gradient_method = gradient_method
        self.noise_probability = noise_probability
        self.circuit_executions = 0
        generator = torch.Generator().manual_seed(seed)
        self.conv_weights = nn.Parameter(
            torch.randn(self.stages, 15, generator=generator, dtype=torch.float64) * init_scale
        )
        self.pool_weights = nn.Parameter(
            torch.randn(self.stages, 3, generator=generator, dtype=torch.float64) * init_scale
        )
        self.output_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.output_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.device = qml.device(backend, wires=qubits)

        @qml.qnode(self.device, interface="torch", diff_method=gradient_method)
        def circuit(
            features: torch.Tensor,
            conv_weights: torch.Tensor,
            pool_weights: torch.Tensor,
        ) -> torch.Tensor:
            for wire in range(qubits):
                qml.RY(features[wire], wires=wire)
            active = list(range(qubits))
            for stage in range(self.stages):
                for pair in convolution_pairs(active):
                    convolution_block(conv_weights[stage], pair)
                    self._apply_noise(pair)
                for source, sink in pooling_pairs(active):
                    pooling_block(pool_weights[stage], source, sink)
                    self._apply_noise((source, sink))
                active = active[1::2]
            return qml.expval(qml.PauliZ(active[0]))

        self.qnode = qml.set_shots(circuit, shots=shots) if shots is not None else circuit

    def _apply_noise(self, wires: tuple[int, int]) -> None:
        if self.noise_probability:
            for wire in wires:
                qml.DepolarizingChannel(self.noise_probability, wires=wire)

    @property
    def trainable_quantum_parameters(self) -> int:
        return self.stages * (15 + 3)

    @property
    def trainable_total_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def reset_execution_count(self) -> None:
        self.circuit_executions = 0

    def quantum_expectation(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 1:
            if features.shape[0] != self.qubits:
                raise ValueError("feature count must equal qubit count")
            self.circuit_executions += 1
            return self.qnode(features, self.conv_weights, self.pool_weights)
        if features.ndim != 2 or features.shape[1] != self.qubits:
            raise ValueError("features must have shape [samples, qubits]")
        outputs = []
        for sample in features:
            self.circuit_executions += 1
            outputs.append(self.qnode(sample, self.conv_weights, self.pool_weights))
        return torch.stack(outputs)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expectations = self.quantum_expectation(features.to(dtype=torch.float64))
        return self.output_scale * expectations + self.output_bias

    def resource_summary(self) -> QCNNResources:
        features = torch.zeros(self.qubits, dtype=torch.float64)
        specs = qml.specs(self.qnode, level="device")(
            features, self.conv_weights.detach(), self.pool_weights.detach()
        )
        resources = specs.resources
        gate_types = {str(name): int(count) for name, count in resources.gate_types.items()}
        two_qubit = int(resources.gate_sizes.get(2, 0))
        return QCNNResources(
            qubits=self.qubits,
            depth=int(resources.depth),
            total_gates=int(resources.num_gates),
            two_qubit_gates=two_qubit,
            gate_types=gate_types,
            trainable_quantum_parameters=self.trainable_quantum_parameters,
            trainable_total_parameters=self.trainable_total_parameters,
        )

    def draw_text(self) -> str:
        drawer = qml.draw(self.qnode, level="device", decimals=2)
        features = torch.zeros(self.qubits, dtype=torch.float64)
        return drawer(features, self.conv_weights.detach(), self.pool_weights.detach())
