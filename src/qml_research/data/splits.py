"""Deterministic stratified sampling and shared cross-validation folds."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def stratified_subsample(y: np.ndarray, sample_size: int | str, seed: int) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    if sample_size == "full" or sample_size == len(y):
        return np.arange(len(y), dtype=int)
    if not isinstance(sample_size, int) or sample_size < 2:
        raise ValueError("sample_size must be 'full' or an integer of at least 2")
    if sample_size > len(y):
        raise ValueError(f"sample_size {sample_size} exceeds dataset size {len(y)}")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=seed)
    selected, _ = next(splitter.split(np.zeros(len(y)), y))
    return np.sort(selected.astype(int))


def make_stratified_folds(
    y: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y, dtype=int)
    class_counts = np.bincount(y, minlength=2)
    if class_counts.min() < n_splits:
        raise ValueError(
            f"each class needs at least {n_splits} samples; observed {class_counts.tolist()}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        (train.astype(int), test.astype(int)) for train, test in splitter.split(np.zeros(len(y)), y)
    ]
