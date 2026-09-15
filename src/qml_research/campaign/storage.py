"""Atomic local artifacts and cooperative interruption."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "results/campaign"


class Paused(RuntimeError):
    """A checkpoint-safe user-requested pause."""


def check_stop() -> None:
    if (OUTPUT / "STOP").exists() or (ROOT / "results/laptop/STOP").exists():
        raise Paused("Stop requested; rerun the same campaign command to resume.")


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:20]


def array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for value in arrays:
        value = np.ascontiguousarray(value)
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def replace_file(source: Path, destination: Path) -> None:
    """Preserve the previous artifact while Windows readers briefly hold the file."""
    for attempt in range(9):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 8:
                raise
            time.sleep(min(0.025 * 2**attempt, 0.5))


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=str, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temp, path)


def atomic_torch(path: Path, value: object) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temp, path)
