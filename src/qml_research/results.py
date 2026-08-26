"""Stable, resumable machine-readable experiment evidence."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def git_state(cwd: str | Path = ".") -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        commit = None
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except subprocess.CalledProcessError:
        branch = None
        dirty = None
    return {"commit": commit, "branch": branch, "dirty": dirty}


class ResultStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def record_path(self, run_id: str) -> Path:
        return self.root / f"{run_id}.json"

    def exists(self, run_id: str) -> bool:
        path = self.record_path(run_id)
        if not path.exists():
            return False
        record = json.loads(path.read_text(encoding="utf-8"))
        return record.get("status") == "completed"

    def write(self, run_id: str, record: dict[str, Any]) -> Path:
        payload = {
            "schema_version": 1,
            "run_id": run_id,
            "recorded_at": datetime.now(UTC).isoformat(),
            **record,
        }
        path = self.record_path(run_id)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(path)
        return path

    def write_kernel_matrix(self, run_id: str, train: np.ndarray, test: np.ndarray) -> Path:
        path = self.root / f"{run_id}.kernel.npz"
        np.savez_compressed(path, train=np.asarray(train), test=np.asarray(test))
        return path


def load_records(root: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(Path(root).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid result JSON: {path}") from exc
        records.append(record)
    return records


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        run = record.get("run", {})
        metrics = record.get("metrics", {})
        resources = record.get("quantum_resources", {})
        kernel_quality = record.get("kernel_quality", {})
        fit = record.get("fit", {})
        rows.append(
            {
                "run_id": record.get("run_id"),
                "status": record.get("status"),
                "experiment": record.get("experiment"),
                "dataset": run.get("dataset"),
                "sample_size": run.get("sample_size"),
                "seed": run.get("seed"),
                "fold": run.get("fold"),
                "model": run.get("model"),
                "reduction": run.get("reduction"),
                "feature_count": run.get("feature_count"),
                "feature_map": run.get("feature_map"),
                "vqc_layers": run.get("vqc_layers"),
                "quantum_condition": run.get("quantum_condition"),
                "preprocessing": run.get("preprocessing"),
                "backend": resources.get("backend"),
                "shots": resources.get("shots"),
                "noise_strength": resources.get("noise_strength"),
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "precision": metrics.get("precision"),
                "sensitivity": metrics.get("sensitivity"),
                "specificity": metrics.get("specificity"),
                "f1": metrics.get("f1"),
                "mcc": metrics.get("mcc"),
                "auroc": metrics.get("auroc"),
                "auprc": metrics.get("auprc"),
                "training_time_seconds": metrics.get("training_time_seconds"),
                "prediction_time_seconds": metrics.get("prediction_time_seconds"),
                "circuit_executions": resources.get("circuit_executions"),
                "num_qubits": resources.get("num_qubits"),
                "circuit_depth": resources.get("circuit_depth"),
                "total_gate_count": resources.get("total_gate_count"),
                "two_qubit_gate_count": resources.get("two_qubit_gate_count"),
                "trainable_parameters": resources.get("trainable_parameters"),
                "kernel_symmetric_error": kernel_quality.get("symmetric_error"),
                "kernel_diagonal_max_error": kernel_quality.get("diagonal_max_error"),
                "kernel_minimum_eigenvalue": kernel_quality.get("minimum_eigenvalue"),
                "kernel_negative_eigenvalues": kernel_quality.get("negative_eigenvalues"),
                "converged": fit.get("converged"),
                "error": record.get("error"),
                "skip_reason": record.get("skip_reason"),
                "projected_runtime_seconds": record.get("projected_runtime_seconds"),
            }
        )
    return pd.DataFrame(rows)
