"""Matched classical, optimized quantum-kernel, and hybrid QMLP models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pennylane as qml
import torch
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn


def classical_model(name: str, seed: int) -> ClassifierMixin:
    models: dict[str, ClassifierMixin] = {
        "dummy": DummyClassifier(strategy="prior", random_state=seed),
        "logistic": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
        "linear_svm": SVC(kernel="linear", class_weight="balanced", probability=False),
        "rbf_svm": SVC(kernel="rbf", class_weight="balanced", probability=False),
        "knn": KNeighborsClassifier(n_neighbors=7),
        "gaussian_nb": GaussianNB(),
        "decision_tree": DecisionTreeClassifier(
            max_depth=4, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=seed, n_jobs=1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=250, class_weight="balanced", random_state=seed, n_jobs=1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=150, random_state=seed),
        "classical_mlp": MLPClassifier(
            hidden_layer_sizes=(16, 8),
            early_stopping=True,
            max_iter=400,
            random_state=seed,
        ),
    }
    try:
        return models[name]
    except KeyError as exc:
        raise ValueError(f"unsupported classical model: {name}") from exc


def decision_scores(model: ClassifierMixin, values: np.ndarray) -> np.ndarray:
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(values), dtype=float).reshape(-1)
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(values), dtype=float)[:, 1]
    return np.asarray(model.predict(values), dtype=float)


class PauliStatevectorKernel:
    """Exact local Pauli Z/ZZ feature-map fidelity kernel.

    This is implemented with PennyLane because Qiskit Machine Learning is absent from the
    project environment. It follows the requested Pauli-style statevector design but is not
    represented as an exact Qiskit API reproduction.
    """

    def __init__(
        self,
        n_qubits: int,
        *,
        reps: int = 2,
        entanglement: str = "linear",
        seed: int = 42,
    ) -> None:
        if n_qubits not in {4, 6, 8, 10}:
            raise ValueError("Pauli kernel supports 4, 6, 8, or 10 qubits")
        if reps not in {1, 2, 3}:
            raise ValueError("Pauli feature-map reps must be 1, 2, or 3")
        if entanglement not in {"linear", "circular", "full"}:
            raise ValueError("unsupported Pauli feature-map entanglement")
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.execution_count = 0
        self.device = qml.device("default.qubit", wires=n_qubits, seed=seed)

        @qml.qnode(self.device)
        def overlap(left: np.ndarray, right: np.ndarray) -> Any:
            self._embed(left)
            qml.adjoint(self._embed)(right)
            return qml.probs(wires=range(n_qubits))

        self.qnode = overlap

    def _pairs(self) -> list[tuple[int, int]]:
        if self.entanglement == "full":
            return [
                (left, right)
                for left in range(self.n_qubits)
                for right in range(left + 1, self.n_qubits)
            ]
        pairs = [(wire, wire + 1) for wire in range(self.n_qubits - 1)]
        if self.entanglement == "circular" and self.n_qubits > 2:
            pairs.append((self.n_qubits - 1, 0))
        return pairs

    def _embed(self, values: np.ndarray) -> None:
        for _ in range(self.reps):
            for wire in range(self.n_qubits):
                qml.Hadamard(wires=wire)
                qml.RZ(2.0 * values[wire], wires=wire)
            for left, right in self._pairs():
                qml.CNOT(wires=(left, right))
                angle = 2.0 * (np.pi - values[left]) * (np.pi - values[right])
                qml.RZ(angle, wires=right)
                qml.CNOT(wires=(left, right))

    def evaluate(self, left: np.ndarray, right: np.ndarray) -> float:
        probability = self.qnode(np.asarray(left, dtype=float), np.asarray(right, dtype=float))
        self.execution_count += 1
        return float(np.asarray(probability)[0])

    def matrix(self, left: np.ndarray, right: np.ndarray, *, symmetric: bool = False) -> np.ndarray:
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("kernel inputs must be two-dimensional")
        if left.shape[1] != self.n_qubits or right.shape[1] != self.n_qubits:
            raise ValueError("kernel feature count must match qubit count")
        if symmetric and not np.array_equal(left, right):
            raise ValueError("symmetric kernel evaluation requires identical arrays")
        result = np.empty((len(left), len(right)), dtype=float)
        if symmetric:
            for row in range(len(left)):
                for column in range(row, len(right)):
                    value = self.evaluate(left[row], right[column])
                    result[row, column] = result[column, row] = value
        else:
            for row in range(len(left)):
                for column in range(len(right)):
                    result[row, column] = self.evaluate(left[row], right[column])
        return result

    def resource_summary(self) -> dict[str, Any]:
        sample = np.zeros(self.n_qubits, dtype=float)
        specs = qml.specs(self.qnode, level="device")(sample, sample)
        resources = specs.resources
        return {
            "qubits": self.n_qubits,
            "feature_map": "pauli_z_zz",
            "reps": self.reps,
            "entanglement": self.entanglement,
            "backend": "pennylane_default.qubit_statevector",
            "shots": None,
            "depth": int(resources.depth),
            "one_qubit_gates": int(resources.gate_sizes.get(1, 0)),
            "entangling_gates": int(resources.gate_sizes.get(2, 0)),
            "total_gates": int(resources.num_gates),
        }


class HybridQMLP(nn.Module):
    """Small classical projection, variational circuit, and classical logit head."""

    def __init__(
        self,
        input_features: int,
        *,
        qubits: int,
        layers: int,
        ansatz: str = "ry_rz_cnot",
        entanglement: str = "linear",
        dropout: float = 0.0,
        init_scale: float = 0.05,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if qubits not in {4, 6, 8, 10} or layers not in {1, 2, 3}:
            raise ValueError("unsupported QMLP qubit/layer configuration")
        if ansatz not in {"ry_rz_cnot", "u3_cnot"}:
            raise ValueError("unsupported QMLP ansatz")
        if entanglement not in {"linear", "circular"}:
            raise ValueError("unsupported QMLP entanglement")
        torch.manual_seed(seed)
        self.input_features = input_features
        self.qubits = qubits
        self.layers = layers
        self.ansatz = ansatz
        self.entanglement = entanglement
        parameters_per_wire = 3 if ansatz == "u3_cnot" else 2
        self.projection = nn.Linear(input_features, qubits, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.quantum_weights = nn.Parameter(
            torch.randn(layers, qubits, parameters_per_wire, dtype=torch.float64) * init_scale
        )
        self.head = nn.Linear(qubits, 1, dtype=torch.float64)
        self.device = qml.device("default.qubit", wires=qubits)
        self.forward_circuit_evaluations = 0

        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit(features: torch.Tensor, weights: torch.Tensor) -> tuple[Any, ...]:
            for wire in range(qubits):
                qml.RY(features[..., wire], wires=wire)
            for layer in range(layers):
                for wire in range(qubits):
                    if ansatz == "u3_cnot":
                        qml.Rot(*weights[layer, wire], wires=wire)
                    else:
                        qml.RY(weights[layer, wire, 0], wires=wire)
                        qml.RZ(weights[layer, wire, 1], wires=wire)
                for wire in range(qubits - 1):
                    qml.CNOT(wires=(wire, wire + 1))
                if entanglement == "circular" and qubits > 2:
                    qml.CNOT(wires=(qubits - 1, 0))
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(qubits))

        self.qnode = circuit

    def quantum_expectations(self, values: torch.Tensor) -> torch.Tensor:
        self.forward_circuit_evaluations += len(values)
        return torch.stack(list(self.qnode(values, self.quantum_weights)), dim=-1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.to(dtype=torch.float64)
        latent = torch.tanh(self.projection(self.dropout(values))) * np.pi
        expectations = self.quantum_expectations(latent)
        return self.head(expectations).squeeze(-1)

    def resource_summary(self) -> dict[str, Any]:
        sample = torch.zeros(self.qubits, dtype=torch.float64)
        specs = qml.specs(self.qnode, level="device")(sample, self.quantum_weights.detach())
        resources = specs.resources
        quantum_parameters = int(self.quantum_weights.numel())
        total = int(sum(parameter.numel() for parameter in self.parameters()))
        return {
            "qubits": self.qubits,
            "layers": self.layers,
            "ansatz": self.ansatz,
            "entanglement": self.entanglement,
            "gradient_method": "backprop_statevector",
            "classical_parameters": total - quantum_parameters,
            "quantum_parameters": quantum_parameters,
            "total_trainable_parameters": total,
            "depth": int(resources.depth),
            "one_qubit_gates": int(resources.gate_sizes.get(1, 0)),
            "entangling_gates": int(resources.gate_sizes.get(2, 0)),
            "total_gates": int(resources.num_gates),
        }


@dataclass
class QMLPTrainingResult:
    best_epoch: int
    optimizer_steps: int
    forward_circuit_evaluations: int
    gradient_evaluations: int
    training_time_seconds: float
    validation_losses: list[float]


def train_qmlp(
    model: HybridQMLP,
    train_values: np.ndarray,
    train_target: np.ndarray,
    validation_values: np.ndarray,
    validation_target: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    loss_name: str,
    patience: int,
) -> QMLPTrainingResult:
    x_train = torch.as_tensor(train_values, dtype=torch.float64)
    y_train = torch.as_tensor(train_target, dtype=torch.float64)
    x_validation = torch.as_tensor(validation_values, dtype=torch.float64)
    y_validation = torch.as_tensor(validation_target, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    positives = max(int(train_target.sum()), 1)
    negatives = max(len(train_target) - positives, 1)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, dtype=torch.float64))
    mse = nn.MSELoss()
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    validation_losses: list[float] = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = (
            bce(logits, y_train)
            if loss_name == "bce_logits"
            else mse(torch.sigmoid(logits), y_train)
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_logits = model(x_validation)
            validation_loss = (
                bce(validation_logits, y_validation)
                if loss_name == "bce_logits"
                else mse(torch.sigmoid(validation_logits), y_validation)
            )
        value = float(validation_loss.item())
        validation_losses.append(value)
        if value < best_loss - 1e-8:
            best_loss = value
            best_epoch = epoch
            best_state = {
                key: tensor.detach().clone() for key, tensor in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    completed_epochs = len(validation_losses)
    return QMLPTrainingResult(
        best_epoch=best_epoch,
        optimizer_steps=completed_epochs,
        forward_circuit_evaluations=model.forward_circuit_evaluations,
        gradient_evaluations=completed_epochs,
        training_time_seconds=elapsed,
        validation_losses=validation_losses,
    )


def qmlp_scores(model: HybridQMLP, values: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(values, dtype=torch.float64)).detach().cpu().numpy()
