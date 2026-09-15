"""Mini-batch training with exact optimizer/RNG/cursor resume checkpoints."""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import pennylane as qml
import torch
from torch import nn

from qml_research.campaign.storage import (
    array_hash,
    atomic_json,
    atomic_torch,
    check_stop,
    fingerprint,
)
from qml_research.ehr_ihd.models import HybridQMLP
from qml_research.qcnn.quantum import HierarchicalQCNN


class ReuploadingVQC(nn.Module):
    """Trainable input scales and repeated encoding, with an entanglement ablation."""

    def __init__(self, features: int, layers: int, seed: int, entangle: bool = True):
        super().__init__()
        torch.manual_seed(seed)
        self.scales = nn.Parameter(torch.ones(layers, features, dtype=torch.float64))
        self.weights = nn.Parameter(torch.randn(layers, features, 2, dtype=torch.float64) * 0.05)
        self.head = nn.Linear(features, 1, dtype=torch.float64)
        device = qml.device("default.qubit", wires=features)

        @qml.qnode(device, interface="torch", diff_method="backprop")
        def circuit(x, weights, scales):
            for layer in range(layers):
                for wire in range(features):
                    qml.RY(x[..., wire] * scales[layer, wire], wires=wire)
                    qml.RY(weights[layer, wire, 0], wires=wire)
                    qml.RZ(weights[layer, wire, 1], wires=wire)
                if entangle:
                    for wire in range(features - 1):
                        qml.CNOT(wires=[wire, wire + 1])
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(features))

        self.circuit = circuit

    def forward(self, x):
        values = torch.stack(self.circuit(x, self.weights, self.scales), dim=-1)
        return self.head(values).squeeze(-1)


def make_model(family: str, features: int, spec: dict, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if family == "hqmlp":
        return HybridQMLP(
            features,
            qubits=features,
            layers=spec["layers"],
            dropout=spec.get("dropout", 0),
            seed=seed,
        )
    if family in {"reupload_vqc", "separable_vqc"}:
        return ReuploadingVQC(features, spec["layers"], seed, family != "separable_vqc")
    if family == "qcnn":
        return HierarchicalQCNN(
            features,
            backend="lightning.qubit",
            shots=None,
            gradient_method="adjoint",
            init_scale=0.05,
            seed=seed,
        )
    if family == "classical_mlp":
        hidden = spec.get("hidden", 16)
        blocks = []
        for layer in range(spec["layers"]):
            blocks.extend(
                [
                    nn.Linear(features if layer == 0 else hidden, hidden, dtype=torch.float64),
                    nn.Tanh(),
                    nn.Dropout(spec.get("dropout", 0)),
                ]
            )
        return nn.Sequential(*blocks, nn.Linear(hidden, 1, dtype=torch.float64), nn.Flatten(0))
    if family == "classical_cnn":
        from qml_research.qcnn.models import SmallCNN

        return nn.Sequential(nn.Unflatten(1, (1, 28, 28)), SmallCNN().double())
    raise ValueError(family)


def scores(model: nn.Module, values: np.ndarray, batch_size: int = 64) -> np.ndarray:
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        return np.concatenate(
            [
                model(torch.as_tensor(values[i : i + batch_size], dtype=dtype))
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
                for i in range(0, len(values), batch_size)
            ]
        )


def train(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    *,
    spec: dict,
    path: Path,
    seed: int,
    on_batch=None,
) -> dict:
    """Resume exactly at the next mini-batch; only loads self-created local files."""
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = path / "last.pt"
    identity = fingerprint(
        {"data": array_hash(x, y, xv, yv), "spec": spec, "seed": seed, "architecture": str(model)}
    )
    dtype = next(model.parameters()).dtype
    xt = torch.as_tensor(x, dtype=dtype)
    yt = torch.as_tensor(y, dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=spec["lr"], weight_decay=spec.get("weight_decay", 0.0001)
    )
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    state = {
        "identity": identity,
        "epoch": 1,
        "offset": 0,
        "permutation": None,
        "history": [],
        "best_loss": float("inf"),
        "best_epoch": 0,
        "best_state": copy.deepcopy(model.state_dict()),
        "stale": 0,
        "steps": 0,
        "elapsed": 0.0,
        "loss_sum": 0.0,
        "seen": 0,
        "completed": False,
    }
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"] != identity:
            raise ValueError(
                "Resume identity mismatch: data, architecture or training config changed"
            )
        state = saved
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        generator.set_state(state["generator"])
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        print(
            f"Resumed {path.name}: epoch {state['epoch']}, "
            f"sample cursor {state['offset']}, steps {state['steps']}",
            flush=True,
        )
    previous_elapsed = state["elapsed"]
    started = time.perf_counter()

    def save():
        state.update(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            generator=generator.get_state(),
            torch_rng=torch.get_rng_state(),
            python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            elapsed=previous_elapsed + time.perf_counter() - started,
        )
        atomic_torch(checkpoint, state)
        atomic_json(
            path / "progress.json",
            {
                k: state[k]
                for k in [
                    "epoch",
                    "offset",
                    "best_epoch",
                    "steps",
                    "elapsed",
                    "completed",
                    "history",
                ]
            },
        )

    positive_weight = torch.tensor((len(y) - y.sum()) / max(y.sum(), 1), dtype=dtype)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    if not checkpoint.exists():
        save()
    while state["epoch"] <= spec["epochs"] and not state["completed"]:
        check_stop()
        if state["permutation"] is None:
            state["permutation"] = torch.randperm(len(x), generator=generator)
        model.train()
        while state["offset"] < len(x):
            check_stop()
            batch = state["permutation"][state["offset"] : state["offset"] + spec["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xt[batch]).reshape(-1), yt[batch])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            state["loss_sum"] += float(loss.detach()) * len(batch)
            state["seen"] += len(batch)
            state["offset"] += len(batch)
            state["steps"] += 1
            save()
            if on_batch is not None:
                on_batch(state["steps"])
        validation_logits = torch.as_tensor(scores(model, xv), dtype=dtype)
        val_loss = float(criterion(validation_logits, torch.as_tensor(yv, dtype=dtype)))
        if not np.isfinite(val_loss):
            raise FloatingPointError("Non-finite validation loss")
        state["history"].append(
            {
                "epoch": state["epoch"],
                "train_loss": state["loss_sum"] / state["seen"],
                "validation_loss": val_loss,
            }
        )
        if val_loss < state["best_loss"] - 1e-8:
            state.update(
                best_loss=val_loss,
                best_epoch=state["epoch"],
                stale=0,
                best_state=copy.deepcopy(model.state_dict()),
            )
            atomic_torch(
                path / "best.pt",
                {
                    "identity": identity,
                    "model": state["best_state"],
                    "epoch": state["best_epoch"],
                    "spec": spec,
                },
            )
        else:
            state["stale"] += 1
        print(
            f"{path.name} epoch {state['epoch']}/{spec['epochs']} validation_loss={val_loss:.6f}",
            flush=True,
        )
        state["completed"] = state["stale"] >= spec["patience"] or state["epoch"] == spec["epochs"]
        state["stopping_reason"] = (
            "early_stopping" if state["stale"] >= spec["patience"] else "epoch_limit"
        )
        state.update(epoch=state["epoch"] + 1, offset=0, permutation=None, loss_sum=0.0, seen=0)
        save()
    model.load_state_dict(state["best_state"])
    return {k: state[k] for k in ["best_epoch", "steps", "elapsed", "history", "stopping_reason"]}
