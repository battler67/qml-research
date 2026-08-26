"""PennyLane state-overlap quantum kernels with auditable execution counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pennylane as qml


@dataclass(frozen=True)
class KernelQuality:
    symmetric_error: float
    diagonal_max_error: float
    minimum_eigenvalue: float
    negative_eigenvalues: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "symmetric_error": self.symmetric_error,
            "diagonal_max_error": self.diagonal_max_error,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "negative_eigenvalues": self.negative_eigenvalues,
        }


class QuantumKernel:
    """Evaluate ``|<phi(x2)|phi(x1)>|^2`` through compute/uncompute circuits."""

    def __init__(
        self,
        n_qubits: int,
        feature_map: str = "angle",
        shots: int | None = None,
        noise_strength: float = 0.0,
        backend: str = "default.qubit",
        seed: int = 42,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if feature_map not in {"angle", "iqp"}:
            raise ValueError(f"unsupported feature map: {feature_map}")
        if not 0.0 <= noise_strength < 1.0:
            raise ValueError("noise_strength must be in [0, 1)")
        if backend != "default.qubit":
            raise ValueError("Phase 1 analytic kernels support default.qubit")
        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.shots = shots
        self.noise_strength = noise_strength
        self.seed = seed
        device_name = "default.mixed" if noise_strength else backend
        self.device_name = device_name
        self.device = qml.device(device_name, wires=n_qubits, seed=seed)
        self.execution_count = 0

        @qml.qnode(self.device)
        def overlap(x1: np.ndarray, x2: np.ndarray) -> Any:
            self._feature_circuit(x1)
            self._noise_layer()
            qml.adjoint(self._feature_circuit)(x2)
            self._noise_layer()
            return qml.probs(wires=range(self.n_qubits))

        self._qnode = qml.set_shots(overlap, shots=shots) if shots is not None else overlap

    def _feature_circuit(self, x: np.ndarray) -> None:
        if self.feature_map == "angle":
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")
        else:
            qml.IQPEmbedding(x, wires=range(self.n_qubits), n_repeats=1)

    def _noise_layer(self) -> None:
        if self.noise_strength:
            for wire in range(self.n_qubits):
                qml.DepolarizingChannel(self.noise_strength, wires=wire)

    def evaluate(self, x1: np.ndarray, x2: np.ndarray) -> float:
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        if x1.shape != (self.n_qubits,) or x2.shape != (self.n_qubits,):
            raise ValueError(f"kernel inputs must each have shape ({self.n_qubits},)")
        probabilities = self._qnode(x1, x2)
        self.execution_count += 1
        return float(np.asarray(probabilities)[0])

    def matrix(
        self,
        left: np.ndarray,
        right: np.ndarray,
        *,
        symmetric: bool = False,
    ) -> np.ndarray:
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("kernel matrix inputs must be two dimensional")
        if left.shape[1] != self.n_qubits or right.shape[1] != self.n_qubits:
            raise ValueError("kernel matrix feature count must equal n_qubits")
        if symmetric and (left.shape != right.shape or not np.array_equal(left, right)):
            raise ValueError("symmetric=True requires identical inputs")

        matrix = np.empty((len(left), len(right)), dtype=float)
        if symmetric:
            for row in range(len(left)):
                for column in range(row, len(right)):
                    value = self.evaluate(left[row], right[column])
                    matrix[row, column] = value
                    matrix[column, row] = value
        else:
            for row, x1 in enumerate(left):
                for column, x2 in enumerate(right):
                    matrix[row, column] = self.evaluate(x1, x2)
        return matrix

    @staticmethod
    def quality(matrix: np.ndarray, tolerance: float = 1e-8) -> KernelQuality:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("quality requires a square Gram matrix")
        symmetric = (matrix + matrix.T) / 2
        eigenvalues = np.linalg.eigvalsh(symmetric)
        return KernelQuality(
            symmetric_error=float(np.max(np.abs(matrix - matrix.T))),
            diagonal_max_error=float(np.max(np.abs(np.diag(matrix) - 1.0))),
            minimum_eigenvalue=float(eigenvalues.min()),
            negative_eigenvalues=int(np.sum(eigenvalues < -tolerance)),
        )

    def resources(self, example: np.ndarray) -> dict[str, Any]:
        try:
            specs = qml.specs(self._qnode, level="device")(
                np.asarray(example, dtype=float), np.asarray(example, dtype=float)
            )
            resources = specs["resources"]
            gate_sizes = dict(resources.gate_sizes)
            return {
                "num_qubits": int(resources.num_allocs),
                "circuit_depth": int(resources.depth),
                "total_gate_count": int(resources.num_gates),
                "two_qubit_gate_count": int(gate_sizes.get(2, 0)),
                "gate_types": dict(resources.gate_types),
            }
        except (KeyError, TypeError, AttributeError):
            return {
                "num_qubits": self.n_qubits,
                "circuit_depth": None,
                "total_gate_count": None,
                "two_qubit_gate_count": None,
                "gate_types": {},
            }

    def draw(self, example: np.ndarray) -> str:
        return qml.draw(self._qnode)(
            np.asarray(example, dtype=float), np.asarray(example, dtype=float)
        )
