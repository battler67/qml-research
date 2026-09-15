"""Dataset auditing and identical, group-safe full-data evaluation splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from qml_research.campaign.storage import OUTPUT, ROOT, array_hash, atomic_json
from qml_research.data.loaders import load_heart, load_wdbc


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    categories: list[str]
    provenance: dict
    official: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    @property
    def groups(self) -> np.ndarray:
        # Identical feature rows remain together even if their labels conflict.
        return pd.util.hash_pandas_object(self.X, index=False).to_numpy()

    @property
    def digest(self) -> str:
        return array_hash(self.X.to_numpy(dtype=float), self.y)


def load_dataset(name: str) -> Dataset:
    if name in {"wdbc", "cleveland"}:
        bundle = load_wdbc() if name == "wdbc" else load_heart(ROOT / "data/raw")
        data = Dataset(
            name, bundle.X, bundle.y, list(bundle.categorical_columns), bundle.provenance
        )
    elif name == "framingham_kaggle":
        root = ROOT / "data/raw/framingham_kaggle"
        frame = pd.read_csv(root / "framingham.csv")
        expected = {
            "male",
            "age",
            "education",
            "currentSmoker",
            "cigsPerDay",
            "BPMeds",
            "prevalentStroke",
            "prevalentHyp",
            "diabetes",
            "totChol",
            "sysBP",
            "diaBP",
            "BMI",
            "heartRate",
            "glucose",
            "TenYearCHD",
        }
        if set(frame.columns) != expected:
            raise ValueError("Unexpected Framingham mirror schema")
        y = frame.pop("TenYearCHD").to_numpy(dtype=int)
        provenance = json.loads((root / "download-manifest.json").read_text())
        verify_download(root, provenance)
        provenance.update(
            endpoint="Kaggle TenYearCHD label; follow-up/censoring cannot be reconstructed",
            validity="Exploratory mirror only; no participant IDs, no verified official linkage",
            license="Unknown in Kaggle API; do not represent as an official licensed release",
            official_teaching_data="Unavailable; requires BioLINCC request; never substituted",
        )
        data = Dataset(name, frame, y, ["education"], provenance)
    elif name == "pima_kaggle":
        root = ROOT / "data/raw/pima_kaggle"
        frame = pd.read_csv(
            root / "pima-indians-diabetes.csv",
            header=None,
            names=[
                "pregnancies",
                "glucose",
                "blood_pressure",
                "skin_thickness",
                "insulin",
                "BMI",
                "pedigree",
                "age",
                "target",
            ],
        )
        y = frame.pop("target").to_numpy(dtype=int)
        columns = ["glucose", "blood_pressure", "skin_thickness", "insulin", "BMI"]
        zero_counts = {column: int((frame[column] == 0).sum()) for column in columns}
        frame[columns] = frame[columns].replace(0, np.nan)
        provenance = json.loads((root / "download-manifest.json").read_text())
        verify_download(root, provenance)
        crosscheck = root / "crosscheck.json"
        if crosscheck.exists():
            provenance["independent_reference_crosscheck"] = json.loads(crosscheck.read_text())
        provenance.update(
            endpoint="Diabetes status; no reconstructed future-risk endpoint",
            reference="https://www.openml.org/api/v1/json/data/37",
            zero_as_missing=zero_counts,
            scope="768 adult women of Pima heritage; limited generalizability",
        )
        data = Dataset(name, frame, y, [], provenance)
    elif name == "breastmnist":
        archive = np.load(ROOT / "data/raw/medmnist/breastmnist.npz")
        counts = [len(archive[f"{part}_images"]) for part in ["train", "val", "test"]]
        images = np.concatenate([archive[f"{part}_images"] for part in ["train", "val", "test"]])
        y = 1 - np.concatenate(
            [archive[f"{part}_labels"].reshape(-1) for part in ["train", "val", "test"]]
        )
        bounds = np.cumsum([0] + counts)
        official = tuple(np.arange(bounds[i], bounds[i + 1]) for i in range(3))
        data = Dataset(
            name,
            pd.DataFrame(images.reshape(len(images), -1) / 255.0),
            y,
            [],
            {
                "source": "https://medmnist.com/",
                "official_counts": counts,
                "endpoint": "Malignant versus normal/benign ultrasound; educational benchmark",
            },
            official,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")
    if set(np.unique(data.y)) != {0, 1} or len(data.X) != len(data.y):
        raise ValueError("Invalid binary dataset")
    if np.isinf(data.X.to_numpy(dtype=float)).any():
        raise ValueError("Infinite input values")
    audit = {
        "dataset": name,
        "rows": len(data.y),
        "features": len(data.X.columns),
        "positive": int(data.y.sum()),
        "negative": int((data.y == 0).sum()),
        "missing": {str(k): int(v) for k, v in data.X.isna().sum().items() if v},
        "duplicate_feature_rows": int(data.X.duplicated().sum()),
        "data_hash": data.digest,
        "provenance": data.provenance,
    }
    atomic_json(OUTPUT / "audits" / f"{name}.json", audit)
    return data


def verify_download(root, manifest):
    for entry in manifest["files"]:
        path = root / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Downloaded file checksum mismatch: {path.name}")


def splits(data: Dataset, seed: int, folds: int = 5):
    if data.official is not None:
        yield 0, data.official
        return
    outer = StratifiedGroupKFold(folds, shuffle=True, random_state=seed)
    for fold, (development, test) in enumerate(outer.split(data.X, data.y, data.groups)):
        inner = StratifiedGroupKFold(5, shuffle=True, random_state=seed + 1000 + fold)
        train_local, val_local = next(
            inner.split(data.X.iloc[development], data.y[development], data.groups[development])
        )
        yield fold, (development[train_local], development[val_local], test)


def prepare(data: Dataset, indices: tuple[np.ndarray, np.ndarray, np.ndarray]):
    numeric = [name for name in data.X.columns if name not in data.categories]
    numerical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pipeline = ColumnTransformer(
        [("numeric", numerical, numeric), ("categorical", categorical, data.categories)]
    )
    train, validation, test = indices
    pipeline.fit(data.X.iloc[train])
    arrays = tuple(
        np.asarray(pipeline.transform(data.X.iloc[part]), dtype=np.float64) for part in indices
    )
    return pipeline, arrays


def reduce_features(arrays, dimensions: int, seed: int):
    reducer = PCA(n_components=dimensions, svd_solver="full", random_state=seed).fit(arrays[0])
    reduced = tuple(reducer.transform(values) for values in arrays)
    scaler = StandardScaler().fit(reduced[0])
    return (reducer, scaler), tuple(scaler.transform(values) for values in reduced)
