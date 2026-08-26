"""Authoritative dataset loaders with explicit disease-positive targets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer, load_iris


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    positive_class: str
    task_description: str
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    provenance: dict[str, Any]
    groups: np.ndarray | None = None

    def validate(self) -> None:
        if len(self.X) != len(self.y):
            raise ValueError("feature and target lengths differ")
        labels = set(np.unique(self.y).tolist())
        if labels != {0, 1}:
            raise ValueError(f"expected binary labels 0/1, received {sorted(labels)}")
        if self.X.columns.duplicated().any():
            raise ValueError("dataset contains duplicate feature names")
        declared = set(self.numerical_columns) | set(self.categorical_columns)
        if declared != set(self.X.columns):
            raise ValueError("numerical/categorical declarations do not cover all features")


def _base_provenance(source: str, citation: str, n_samples: int, n_features: int) -> dict[str, Any]:
    return {
        "source": source,
        "citation": citation,
        "n_samples": n_samples,
        "n_features": n_features,
        "sklearn_version": sklearn.__version__,
        "public_non_identifiable": True,
    }


def load_iris_binary() -> DatasetBundle:
    raw = load_iris(as_frame=True)
    frame = raw.frame.iloc[:100].copy()
    target_name = raw.target.name
    X = frame.drop(columns=[target_name]).reset_index(drop=True)
    y = frame[target_name].astype(int).to_numpy()
    bundle = DatasetBundle(
        name="iris",
        X=X,
        y=y,
        positive_class="Iris versicolor (class 1)",
        task_description="Non-medical binary implementation sanity check",
        numerical_columns=tuple(X.columns),
        categorical_columns=(),
        provenance=_base_provenance(
            source="sklearn.datasets.load_iris, first two classes",
            citation="Fisher, R.A. (1936), The use of multiple measurements in taxonomic problems",
            n_samples=len(X),
            n_features=X.shape[1],
        ),
    )
    bundle.validate()
    return bundle


def load_wdbc() -> DatasetBundle:
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy().reset_index(drop=True)
    # sklearn encodes malignant=0 and benign=1. Disease must be the positive class.
    y = (1 - raw.target.astype(int)).to_numpy()
    bundle = DatasetBundle(
        name="wdbc",
        X=X,
        y=y,
        positive_class="Malignant breast mass",
        task_description=(
            "Diagnostic classification from digitized fine-needle-aspirate features; "
            "not longitudinal or pre-symptomatic early detection"
        ),
        numerical_columns=tuple(X.columns),
        categorical_columns=(),
        provenance=_base_provenance(
            source="sklearn.datasets.load_breast_cancer",
            citation=(
                "Wolberg, Street and Mangasarian, Wisconsin Diagnostic Breast Cancer; "
                "UCI ML Repository"
            ),
            n_samples=len(X),
            n_features=X.shape[1],
        ),
    )
    bundle.validate()
    return bundle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_heart(cache_csv: Path, manifest_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=45)
    X = dataset.data.features.copy()
    target = dataset.data.targets.iloc[:, 0].copy()
    frame = X.copy()
    frame["target"] = target
    frame = frame.replace("?", np.nan)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache_csv, index=False)
    metadata = {
        "source": "UCI Machine Learning Repository dataset 45 via ucimlrepo",
        "url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "doi": "10.24432/C52P4X",
        "license": "CC BY 4.0",
        "retrieved_at": datetime.now(UTC).isoformat(),
        "sha256": _sha256(cache_csv),
        "n_samples": len(frame),
        "n_features": len(frame.columns) - 1,
        "public_non_identifiable": True,
    }
    manifest_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return frame, metadata


def load_heart(cache_dir: str | Path = "data/raw") -> DatasetBundle:
    cache_root = Path(cache_dir)
    cache_csv = cache_root / "uci_heart_disease.csv"
    manifest_path = cache_root / "uci_heart_disease.provenance.json"
    if cache_csv.exists() and manifest_path.exists():
        frame = pd.read_csv(cache_csv)
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_hash = _sha256(cache_csv)
        if current_hash != metadata.get("sha256"):
            raise ValueError("cached Heart Disease checksum does not match provenance manifest")
    else:
        frame, metadata = _fetch_heart(cache_csv, manifest_path)

    if "target" not in frame:
        raise ValueError("Heart Disease cache has no target column")
    X = frame.drop(columns=["target"]).copy().reset_index(drop=True)
    y_raw = pd.to_numeric(frame["target"], errors="coerce")
    valid = y_raw.notna()
    X = X.loc[valid].reset_index(drop=True)
    y = (y_raw.loc[valid].to_numpy(dtype=float) > 0).astype(int)

    categorical = ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")
    missing_columns = set(categorical) - set(X.columns)
    if missing_columns:
        raise ValueError(f"Heart Disease source is missing columns: {sorted(missing_columns)}")
    numerical = tuple(column for column in X.columns if column not in categorical)
    metadata = {
        **metadata,
        "target_mapping": "0=no disease, values 1-4=disease present",
        "effective_n_samples": len(X),
    }
    bundle = DatasetBundle(
        name="heart",
        X=X,
        y=y,
        positive_class="Heart disease present (original target > 0)",
        task_description="Cleveland Heart Disease presence-versus-absence classification",
        numerical_columns=numerical,
        categorical_columns=categorical,
        provenance=metadata,
    )
    bundle.validate()
    return bundle


def load_dataset(name: str, cache_dir: str | Path = "data/raw") -> DatasetBundle:
    loaders = {
        "iris": load_iris_binary,
        "wdbc": load_wdbc,
        "heart": lambda: load_heart(cache_dir),
    }
    try:
        return loaders[name]()
    except KeyError as exc:
        raise ValueError(f"unsupported dataset: {name}") from exc
