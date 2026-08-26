"""Shallow PennyLane variational quantum classifier for controlled experiments."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


@dataclass(frozen=True)
class VQCFitSummary:
    training_time_seconds: float
    training_executions: int
    trainable_parameters: int
    converged: bool
    final_loss: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "training_time_seconds": self.training_time_seconds,
            "training_executions": self.training_executions,
            "trainable_parameters": self.trainable_parameters,
            "converged": self.converged,
            "final_loss": self.final_loss,
        }


class VariationalQuantumClassifier:
    """Angle-embedded binary VQC with parameter-shift gradients."""

    def __init__(
        self,
        n_qubits: int,
        layers: int = 1,
        steps: int = 100,
        batch_size: int = 16,
        learning_rate: float = 0.05,
        backend: str = "default.qubit",
        seed: int = 42,
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if layers not in {1, 2}:
            raise ValueError("Phase 1 supports one or two VQC layers")
        if steps < 1 or batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if backend != "default.qubit":
            raise ValueError("Phase 1 VQC supports default.qubit")
        self.n_qubits = n_qubits
        self.layers = layers
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.device_name = backend
        self.device = qml.device(self.device_name, wires=n_qubits, shots=None, seed=seed)

        @qml.qnode(self.device, interface="autograd", diff_method="parameter-shift")
        def circuit(x: np.ndarray, weights: np.ndarray) -> Any:
            qml.AngleEmbedding(x, wires=range(self.n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            return qml.expval(qml.PauliZ(0))

        self._qnode = circuit
        self.parameters_: pnp.ndarray | None = None
        self.loss_history_: list[float] = []
        self.execution_count = 0
        self.fit_summary_: VQCFitSummary | None = None

    @property
    def weight_shape(self) -> tuple[int, int, int]:
        return (self.layers, self.n_qubits, 3)

    @property
    def parameter_count(self) -> int:
        return int(np.prod(self.weight_shape)) + 1

    def _unpack(self, parameters: pnp.ndarray) -> tuple[pnp.ndarray, pnp.ndarray]:
        weights = parameters[:-1].reshape(self.weight_shape)
        return weights, parameters[-1]

    def _loss(
        self,
        parameters: pnp.ndarray,
        X: np.ndarray,
        y: np.ndarray,
    ) -> pnp.ndarray:
        weights, bias = self._unpack(parameters)
        expectations = pnp.stack([self._qnode(row, weights) for row in X])
        probabilities = pnp.clip((expectations + bias + 1.0) / 2.0, 1e-7, 1 - 1e-7)
        targets = pnp.asarray(y, requires_grad=False)
        return -pnp.mean(
            targets * pnp.log(probabilities) + (1 - targets) * pnp.log(1 - probabilities)
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> VariationalQuantumClassifier:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if X.ndim != 2 or X.shape[1] != self.n_qubits:
            raise ValueError(f"X must have shape (n_samples, {self.n_qubits})")
        if len(X) != len(y):
            raise ValueError("X and y lengths differ")
        rng = np.random.default_rng(self.seed)
        initial = np.concatenate(
            [0.05 * rng.standard_normal(self.parameter_count - 1), np.zeros(1, dtype=float)]
        )
        parameters = pnp.array(initial, requires_grad=True)
        optimizer = qml.AdamOptimizer(stepsize=self.learning_rate)
        self.loss_history_ = []
        started = perf_counter()

        with qml.Tracker(self.device) as tracker:
            for _ in range(self.steps):
                batch_n = min(self.batch_size, len(X))
                indices = rng.choice(len(X), size=batch_n, replace=False)
                X_batch = X[indices]
                y_batch = y[indices]

                def objective(
                    candidate: pnp.ndarray,
                    X_batch: np.ndarray = X_batch,
                    y_batch: np.ndarray = y_batch,
                ) -> pnp.ndarray:
                    return self._loss(candidate, X_batch, y_batch)

                parameters, value = optimizer.step_and_cost(objective, parameters)
                self.loss_history_.append(float(value))

        elapsed = perf_counter() - started
        self.parameters_ = parameters
        training_executions = int(tracker.totals.get("executions", 0))
        self.execution_count += training_executions
        recent = self.loss_history_[-min(10, len(self.loss_history_)) :]
        converged = bool(np.isfinite(recent).all() and min(recent) < self.loss_history_[0])
        self.fit_summary_ = VQCFitSummary(
            training_time_seconds=elapsed,
            training_executions=training_executions,
            trainable_parameters=self.parameter_count,
            converged=converged,
            final_loss=float(self.loss_history_[-1]),
        )
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.parameters_ is None:
            raise RuntimeError("fit must be called before prediction")
        X = np.asarray(X, dtype=float)
        weights, bias = self._unpack(self.parameters_)
        with qml.Tracker(self.device) as tracker:
            scores = np.asarray([float(self._qnode(row, weights) + bias) for row in X])
        self.execution_count += int(tracker.totals.get("executions", 0))
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(int)

    def resources(self, example: np.ndarray) -> dict[str, Any]:
        if self.parameters_ is None:
            weights = np.zeros(self.weight_shape, dtype=float)
        else:
            weights, _ = self._unpack(self.parameters_)
        try:
            specs = qml.specs(self._qnode, level="device")(
                np.asarray(example, dtype=float), weights
            )
            resources = specs["resources"]
            gate_sizes = dict(resources.gate_sizes)
            return {
                "num_qubits": int(resources.num_allocs),
                "circuit_depth": int(resources.depth),
                "total_gate_count": int(resources.num_gates),
                "two_qubit_gate_count": int(gate_sizes.get(2, 0)),
                "gate_types": dict(resources.gate_types),
                "trainable_parameters": self.parameter_count,
            }
        except (KeyError, TypeError, AttributeError):
            return {
                "num_qubits": self.n_qubits,
                "circuit_depth": None,
                "total_gate_count": None,
                "two_qubit_gate_count": None,
                "gate_types": {},
                "trainable_parameters": self.parameter_count,
            }

    def draw(self, example: np.ndarray) -> str:
        if self.parameters_ is None:
            weights = np.zeros(self.weight_shape, dtype=float)
        else:
            weights, _ = self._unpack(self.parameters_)
        return qml.draw(self._qnode)(np.asarray(example, dtype=float), weights)
