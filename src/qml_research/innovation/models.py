"""An uncertainty-gated residual; quantum claims require the matched ablations."""

from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from torch import nn


class GatedResidual(nn.Module):
    """Inputs are [frozen classical logit, bounded PCA coordinates]."""

    def __init__(self, qubits, spec, seed, variant="quantum"):
        super().__init__()
        torch.manual_seed(seed)
        self.variant, self.qubits, self.layers = variant, qubits, spec["layers"]
        self.calibration = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float64))
        self.scales = nn.Parameter(torch.ones(self.layers, qubits, dtype=torch.float64))
        self.angles = nn.Parameter(torch.randn(self.layers, qubits, 2, dtype=torch.float64) * 0.1)
        self.head = nn.Linear(qubits, 1, dtype=torch.float64)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        # Approximately match trainable circuit+head parameter count with a tanh MLP.
        hidden = max(1, round((3 * self.layers * qubits + qubits) / (qubits + 2)))
        self.replacement = nn.Sequential(
            nn.Linear(qubits, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=torch.float64),
        )
        for parameter in self.replacement.parameters():
            parameter.requires_grad_(variant == "classical")
        for parameter in [self.scales, self.angles, *self.head.parameters()]:
            parameter.requires_grad_(variant not in {"classical", "no_quantum"})
        if variant == "frozen":
            self.scales.requires_grad_(False)
            self.angles.requires_grad_(False)
        device = qml.device("default.qubit", wires=qubits)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(x, scales, angles):
            for layer in range(self.layers):
                for wire in range(qubits):
                    qml.RY(x[..., wire] * scales[layer, wire], wires=wire)
                    qml.RZ(angles[layer, wire, 0], wires=wire)
                    qml.RY(angles[layer, wire, 1], wires=wire)
                if variant != "separable":
                    for wire in range(qubits - 1):
                        qml.CNOT(wires=[wire, wire + 1])
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(qubits))

        self.circuit = circuit

    def extra_repr(self):
        return f"variant={self.variant}, qubits={self.qubits}, layers={self.layers}"

    def forward(self, values):
        base, x = values[:, 0], values[:, 1:]
        calibrated = self.calibration[0] * base + self.calibration[1]
        if self.variant == "no_quantum":
            return calibrated
        if self.variant == "classical":
            residual = self.replacement(x).squeeze(-1)
        else:
            features = torch.stack(self.circuit(x, self.scales, self.angles), dim=-1)
            residual = self.head(features).squeeze(-1)
        gate = torch.ones_like(base) if self.variant == "ungated" else 1 - torch.tanh(base / 2) ** 2
        return calibrated + gate * residual

    def telemetry(self):
        quantum = self.variant not in {"classical", "no_quantum"}
        return {
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "qubits": self.qubits if quantum else 0,
            "layers": self.layers,
            "backend": "default.qubit" if quantum else "torch",
            "shots": None,
            "gradient": "analytic statevector backpropagation" if quantum else "backpropagation",
            "single_qubit_gates_per_sample": 3 * self.layers * self.qubits if quantum else 0,
            "cnot_per_sample": self.layers * (self.qubits - 1)
            if quantum and self.variant != "separable"
            else 0,
            "quantum_executions_measured": False,
        }


def residual_inputs(original, reduced, anchor):
    return np.column_stack([anchor.decision_function(original), reduced])
