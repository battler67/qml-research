"""Exact small-circuit kernels with resumable state caches and bounded blocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pennylane as qml

from qml_research.campaign.storage import array_hash, atomic_json, check_stop, fingerprint


def embedding(values, qubits: int, kind: str, repeats: int):
    if kind == "angle":
        qml.AngleEmbedding(values, wires=range(qubits), rotation="Y")
    elif kind == "iqp":
        qml.IQPEmbedding(values, wires=range(qubits), n_repeats=repeats)
    else:
        raise ValueError(kind)


def states(
    values: np.ndarray, *, kind: str, repeats: int, path: Path | None = None, block_size: int = 128
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n, qubits = values.shape
    if not 1 <= qubits <= 10:
        raise ValueError("This laptop campaign supports at most ten qubits")
    identity = fingerprint(
        {"values": array_hash(values), "kind": kind, "repeats": repeats, "schema": 1}
    )
    device = qml.device("default.qubit", wires=qubits)

    @qml.qnode(device)
    def circuit(x):
        embedding(x, qubits, kind, repeats)
        return qml.state()

    start = 0
    if path is None:
        output = np.empty((n, 2**qubits), dtype=np.complex128)
    else:
        path.mkdir(parents=True, exist_ok=True)
        metadata = path / "progress.json"
        array_path = path / "states.npy"
        if metadata.exists() and array_path.exists():
            saved = json.loads(metadata.read_text())
            if saved["identity"] != identity:
                raise ValueError("Kernel cache identity mismatch")
            start = saved["rows_completed"]
            output = np.lib.format.open_memmap(array_path, mode="r+")
            if output.shape != (n, 2**qubits) or output.dtype != np.complex128:
                raise ValueError("Kernel cache shape/dtype mismatch")
        else:
            output = np.lib.format.open_memmap(
                array_path, mode="w+", dtype=np.complex128, shape=(n, 2**qubits)
            )
    for begin in range(start, n, block_size):
        check_stop()
        end = min(n, begin + block_size)
        output[begin:end] = np.asarray(circuit(values[begin:end]))
        if path is not None:
            output.flush()
            atomic_json(
                path / "progress.json",
                {
                    "identity": identity,
                    "rows_completed": end,
                    "total_rows": n,
                    "completed": end == n,
                },
            )
    return output


def fidelity(left: np.ndarray, right: np.ndarray, block_size: int = 128) -> np.ndarray:
    # At 4,238 rows the largest full matrix is about 137 MiB, not exponential in rows.
    if len(left) * len(right) * 8 > 512 * 1024**2:
        raise MemoryError("Kernel matrix exceeds this campaign's 512 MiB matrix budget")
    result = np.empty((len(left), len(right)), dtype=np.float64)
    right_adjoint = right.conj().T
    for start in range(0, len(left), block_size):
        check_stop()
        overlap = left[start : start + block_size] @ right_adjoint
        result[start : start + block_size] = np.abs(overlap) ** 2
    return np.clip(result, 0, 1, out=result)


def pauli_projection(state: np.ndarray) -> np.ndarray:
    """All one-qubit X/Y/Z expectations in PennyLane's big-endian wire order."""
    qubits = int(np.log2(state.shape[1]))
    indices = np.arange(state.shape[1])
    features = []
    for wire in range(qubits):
        bit = 1 << (qubits - wire - 1)
        zero = indices[(indices & bit) == 0]
        one = zero | bit
        product = np.sum(state[:, zero].conj() * state[:, one], axis=1)
        features.extend(
            [
                2 * product.real,
                2 * product.imag,
                np.sum(np.abs(state[:, zero]) ** 2 - np.abs(state[:, one]) ** 2, axis=1),
            ]
        )
    return np.stack(features, axis=1)
