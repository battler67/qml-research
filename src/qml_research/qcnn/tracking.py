"""Stable run directories and reproducibility metadata."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def stable_hash(payload: dict[str, Any], length: int = 16) -> str:
    serialized = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:length]


def _git_metadata(cwd: Path) -> dict[str, Any]:
    def command(*arguments: str) -> str | None:
        result = subprocess.run(
            ["git", *arguments], cwd=cwd, capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty": bool(command("status", "--porcelain")),
    }


def system_metadata(cwd: str | Path = ".") -> dict[str, Any]:
    packages = {}
    for package in (
        "numpy",
        "scikit-learn",
        "pennylane",
        "pennylane-lightning",
        "torch",
        "torchvision",
        "medmnist",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    memory: dict[str, Any] = {}
    try:
        import psutil

        virtual = psutil.virtual_memory()
        memory = {"total_bytes": virtual.total, "available_bytes": virtual.available}
    except ImportError:
        memory = {"total_bytes": None, "available_bytes": None}
    gpu: dict[str, Any] = {"available": False}
    try:
        import torch

        gpu = {
            "available": bool(torch.cuda.is_available()),
            "count": int(torch.cuda.device_count()),
            "names": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        pass
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "gpu": gpu,
        "packages": packages,
        "git": _git_metadata(Path(cwd)),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(output)
    return output


class RunDirectory:
    def __init__(self, root: str | Path, identity: dict[str, Any]) -> None:
        self.run_id = stable_hash(identity)
        self.path = Path(root) / "runs" / self.run_id
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def record_path(self) -> Path:
        return self.path / "record.json"

    def completed(self) -> bool:
        if not self.record_path.exists():
            return False
        payload = json.loads(self.record_path.read_text(encoding="utf-8"))
        return payload.get("status") == "completed"

    def write(self, name: str, payload: dict[str, Any]) -> Path:
        return write_json(self.path / name, payload)
