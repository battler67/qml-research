"""Leakage-safe BreastMNIST and WDBC data pipelines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from qml_research.qcnn.config import QCNNConfig

BREASTMNIST_MD5 = "750601b1f35ba3300ea97c75c52ff8f6"
BREASTMNIST_COUNTS = {"train": 546, "validation": 78, "test": 156}
BREASTMNIST_LICENSE = "CC BY 4.0"


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray


@dataclass
class DatasetSplits:
    name: str
    train: Split
    validation: Split
    test: Split
    provenance: dict[str, Any]
    is_image: bool

    def validate(self) -> None:
        for name in ("train", "validation", "test"):
            split = getattr(self, name)
            if len(split.X) != len(split.y) or len(split.X) != len(split.indices):
                raise ValueError(f"{name} data, labels, and indices differ in length")
            labels = set(np.unique(split.y).tolist())
            if not labels.issubset({0, 1}) or len(labels) < 2:
                raise ValueError(f"{name} must contain both binary classes")

    def manifest(self) -> dict[str, Any]:
        result: dict[str, Any] = {"dataset": self.name, "provenance": self.provenance}
        for name in ("train", "validation", "test"):
            split = getattr(self, name)
            values, counts = np.unique(split.y, return_counts=True)
            result[name] = {
                "count": len(split.y),
                "class_counts": {
                    str(int(value)): int(count) for value, count in zip(values, counts, strict=True)
                },
                "indices": split.indices.astype(int).tolist(),
            }
        return result


def file_checksum(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stratified_indices(
    y: np.ndarray,
    limit: int | str,
    seed: int,
    samples_per_class: int | None,
) -> np.ndarray:
    y = np.asarray(y, dtype=int)
    if samples_per_class is not None:
        rng = np.random.default_rng(seed)
        selected: list[int] = []
        for label in (0, 1):
            candidates = np.flatnonzero(y == label)
            if samples_per_class > len(candidates):
                raise ValueError(
                    f"samples_per_class={samples_per_class} exceeds class {label} count"
                )
            selected.extend(rng.choice(candidates, samples_per_class, replace=False).tolist())
        return np.asarray(sorted(selected), dtype=int)
    if limit == "full" or int(limit) == len(y):
        return np.arange(len(y), dtype=int)
    requested = int(limit)
    if requested > len(y):
        raise ValueError(f"requested {requested} samples but split contains {len(y)}")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=requested, random_state=seed)
    selected, _ = next(splitter.split(np.zeros(len(y)), y))
    return np.sort(selected.astype(int))


def _select_split(
    X: np.ndarray,
    y: np.ndarray,
    limit: int | str,
    seed: int,
    samples_per_class: int | None,
) -> Split:
    selected = _stratified_indices(y, limit, seed, samples_per_class)
    return Split(X=np.asarray(X)[selected], y=np.asarray(y)[selected], indices=selected)


def _load_medmnist_split(root: Path, split: str, download: bool) -> tuple[np.ndarray, np.ndarray]:
    try:
        from medmnist import BreastMNIST
    except ImportError as exc:  # pragma: no cover - exercised without optional dependency
        raise RuntimeError('BreastMNIST requires `pip install -e ".[qcnn]"`') from exc
    dataset = BreastMNIST(root=str(root), split=split, download=download, size=28)
    images = np.asarray(dataset.imgs, dtype=np.uint8)
    original_labels = np.asarray(dataset.labels, dtype=int).reshape(-1)
    # Official MedMNIST: 0=malignant, 1=normal/benign. Disease is positive here.
    labels = (original_labels == 0).astype(int)
    return images, labels


def load_breastmnist(config: QCNNConfig, *, download: bool = False) -> DatasetSplits:
    root = Path(config.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / "breastmnist.npz"
    train_X, train_y = _load_medmnist_split(root, "train", download)
    validation_X, validation_y = _load_medmnist_split(root, "val", download)
    test_X, test_y = _load_medmnist_split(root, "test", download)
    if (len(train_y), len(validation_y), len(test_y)) != (546, 78, 156):
        raise ValueError("BreastMNIST official split sizes changed unexpectedly")
    md5 = file_checksum(npz_path, "md5") if npz_path.exists() else None
    if md5 is not None and md5 != BREASTMNIST_MD5:
        raise ValueError(f"BreastMNIST checksum mismatch: {md5}")
    seed = config.seeds[0]
    splits = DatasetSplits(
        name="breastmnist",
        train=_select_split(train_X, train_y, config.train_size, seed, config.samples_per_class),
        validation=_select_split(
            validation_X,
            validation_y,
            config.validation_size,
            seed + 1,
            config.samples_per_class,
        ),
        test=_select_split(test_X, test_y, config.test_size, seed + 2, config.samples_per_class),
        provenance={
            "dataset_version": config.dataset_version,
            "source": "MedMNIST BreastMNIST",
            "source_url": "https://medmnist.com/",
            "license": BREASTMNIST_LICENSE,
            "md5": md5,
            "official_split_counts": BREASTMNIST_COUNTS,
            "original_labels": {"0": "malignant", "1": "normal, benign"},
            "target_mapping": "0=normal/benign, 1=malignant (remapped)",
            "clinical_use": False,
        },
        is_image=True,
    )
    splits.validate()
    return splits


def load_wdbc_fold(config: QCNNConfig, *, seed: int, fold: int = 0) -> DatasetSplits:
    raw = load_breast_cancer()
    X = np.asarray(raw.data, dtype=np.float64)
    y = (1 - np.asarray(raw.target, dtype=int)).astype(int)
    duplicates = int(len(X) - len(np.unique(X, axis=0)))
    outer = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=seed)
    try:
        train_validation_indices, test_indices = list(outer.split(X, y))[fold]
    except IndexError as exc:
        raise ValueError(f"fold {fold} is outside 0..{config.cv_folds - 1}") from exc
    inner = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + fold)
    inner_train, inner_validation = next(
        inner.split(X[train_validation_indices], y[train_validation_indices])
    )
    train_indices = train_validation_indices[inner_train]
    validation_indices = train_validation_indices[inner_validation]

    def selected(indices: np.ndarray, limit: int | str, salt: int) -> Split:
        local = _stratified_indices(y[indices], limit, seed + salt, config.samples_per_class)
        original = np.asarray(indices)[local]
        return Split(X=X[original], y=y[original], indices=original.astype(int))

    splits = DatasetSplits(
        name="wdbc",
        train=selected(train_indices, config.train_size, 10 + fold),
        validation=selected(validation_indices, config.validation_size, 20 + fold),
        test=selected(test_indices, config.test_size, 30 + fold),
        provenance={
            "dataset_version": "sklearn bundled UCI WDBC",
            "source": "sklearn.datasets.load_breast_cancer / UCI dataset 17",
            "source_url": (
                "https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic"
            ),
            "raw_samples": len(X),
            "raw_features": X.shape[1],
            "missing_values": int(np.isnan(X).sum()),
            "duplicate_feature_rows": duplicates,
            "target_mapping": "0=benign, 1=malignant (remapped)",
            "task_boundary": "diagnostic FNA mass classification, not screening",
        },
        is_image=False,
    )
    splits.validate()
    return splits


def load_splits(
    config: QCNNConfig,
    *,
    download: bool = False,
    seed: int | None = None,
    fold: int = 0,
) -> DatasetSplits:
    if config.dataset == "breastmnist":
        return load_breastmnist(config, download=download)
    return load_wdbc_fold(config, seed=seed if seed is not None else config.seeds[0], fold=fold)


class FeatureReducer:
    """Fit/transform reducer whose learned state sees training data only."""

    def __init__(self, method: str, n_features: int, seed: int, *, is_image: bool) -> None:
        self.method = method
        self.n_features = n_features
        self.seed = seed
        self.is_image = is_image
        self.standardizer: StandardScaler | None = None
        self.pca: PCA | None = None
        self.angle_scaler: MinMaxScaler | None = None
        self.fit_sample_count: int | None = None

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64).reshape(len(X), -1)

    def _spatial(self, X: np.ndarray) -> np.ndarray:
        images = np.asarray(X, dtype=np.float64)
        if images.ndim != 3 or images.shape[1:] != (28, 28):
            raise ValueError("spatial_pool expects N x 28 x 28 grayscale images")
        rows, columns = (2, 2) if self.n_features == 4 else (2, 4)
        pooled = images.reshape(len(images), rows, 28 // rows, columns, 28 // columns)
        return pooled.mean(axis=(2, 4)).reshape(len(images), -1) / 255.0

    def fit(self, X: np.ndarray) -> FeatureReducer:
        self.fit_sample_count = len(X)
        if self.method == "spatial_pool":
            if not self.is_image:
                raise ValueError("spatial_pool is only valid for images")
            self._spatial(X)
            return self
        flattened = self._flatten(X)
        self.standardizer = StandardScaler().fit(flattened)
        standardized = self.standardizer.transform(flattened)
        self.pca = PCA(n_components=self.n_features, random_state=self.seed).fit(standardized)
        reduced = self.pca.transform(standardized)
        self.angle_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi), clip=True).fit(reduced)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.fit_sample_count is None:
            raise RuntimeError("FeatureReducer must be fitted before transform")
        if self.method == "spatial_pool":
            return np.pi * (2.0 * self._spatial(X) - 1.0)
        if self.standardizer is None or self.pca is None or self.angle_scaler is None:
            raise RuntimeError("PCA reducer has incomplete fitted state")
        standardized = self.standardizer.transform(self._flatten(X))
        return self.angle_scaler.transform(self.pca.transform(standardized))

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "n_features": self.n_features,
            "fit_sample_count": self.fit_sample_count,
            "explained_variance_ratio": (
                self.pca.explained_variance_ratio_.tolist() if self.pca is not None else None
            ),
        }


def prepare_features(
    splits: DatasetSplits, config: QCNNConfig, seed: int
) -> tuple[dict[str, np.ndarray], FeatureReducer]:
    reducer = FeatureReducer(
        config.reducer, config.reduced_features, seed, is_image=splits.is_image
    )
    features = {
        "train": reducer.fit_transform(splits.train.X),
        "validation": reducer.transform(splits.validation.X),
        "test": reducer.transform(splits.test.X),
    }
    return features, reducer


def write_manifest(splits: DatasetSplits, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(splits.manifest(), indent=2, sort_keys=True), encoding="utf-8")
    return output
