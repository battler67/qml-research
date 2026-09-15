"""Official-source download and development-only preprocessing."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from qml_research.campaign.data import Dataset, prepare
from qml_research.campaign.storage import ROOT, atomic_json, fingerprint

DATA = ROOT / "data/raw/uci_ckd_innovation"
OUTPUT = ROOT / "results/innovation"
NUMERIC = ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wbcc", "rbcc"]
CATEGORIES = ["sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    if (DATA / "manifest.json").exists():
        return
    with urllib.request.urlopen("https://archive.ics.uci.edu/api/dataset?id=336", timeout=60) as r:
        metadata = json.load(r)
    url = metadata["data"]["data_url"]
    if url != "https://archive.ics.uci.edu/static/public/336/data.csv":
        raise ValueError("Official data URL changed; review source before downloading")
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = r.read(2_000_001)
    if len(payload) > 2_000_000:
        raise ValueError("Unexpected dataset size")
    (DATA / "data.csv").write_bytes(payload)
    atomic_json(DATA / "metadata.json", metadata)
    atomic_json(DATA / "manifest.json", {"url": url, "sha256": hashlib.sha256(payload).hexdigest()})


def load():
    payload = (DATA / "data.csv").read_bytes()
    manifest = json.loads((DATA / "manifest.json").read_text())
    if hashlib.sha256(payload).hexdigest() != manifest["sha256"]:
        raise ValueError("CKD checksum changed")
    frame = pd.read_csv(DATA / "data.csv", dtype=str)
    if set(frame) != set(NUMERIC + CATEGORIES + ["class"]) or len(frame) != 400:
        raise ValueError("Unexpected official CKD schema or row count")
    for col in frame:
        frame[col] = frame[col].str.strip().replace({"?": np.nan, "": np.nan})
    target = frame.pop("class")
    if set(target) != {"ckd", "notckd"}:
        raise ValueError("Unexpected CKD target")
    for col in NUMERIC:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if np.isinf(frame[NUMERIC].to_numpy()).any():
        raise ValueError("Infinite input")
    return Dataset(
        "uci_ckd_innovation", frame, (target == "ckd").to_numpy(int), CATEGORIES, manifest
    )


def partition(data: Dataset, seed: int, config: dict):
    # This master holdout is identical across configurations and initialization seeds.
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=9152026)
    dev, test = next(splitter.split(data.X, data.y, data.groups))
    inner = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    a, b = next(inner.split(data.X.iloc[dev], data.y[dev], data.groups[dev]))
    train, val = dev[a], dev[b]
    for name, part in [("train_cap", train), ("validation_cap", val)]:
        if config.get(name) and len(part) > config[name]:
            part, _ = train_test_split(
                part, train_size=config[name], stratify=data.y[part], random_state=seed
            )
        if name == "train_cap":
            train = part
        else:
            val = part
    return train, val, test


def preprocess(data, train, val, qubits):
    # Pass validation twice: prepare() never even transforms the locked test features.
    pipeline, arrays = prepare(data, (train, val, val))
    pca = PCA(n_components=qubits, svd_solver="full").fit(arrays[0])
    scaler = StandardScaler().fit(pca.transform(arrays[0]))

    def reduced(x):
        return np.pi * np.tanh(scaler.transform(pca.transform(x)) / 2)

    return (
        {"pipeline": pipeline, "pca": pca, "scaler": scaler},
        arrays[:2],
        tuple(reduced(x) for x in arrays[:2]),
    )


def transform(saved, frame):
    original = np.asarray(saved["pipeline"].transform(frame), dtype=float)
    reduced = np.pi * np.tanh(saved["scaler"].transform(saved["pca"].transform(original)) / 2)
    return original, reduced


def seal(path: Path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Immutable experiment identity changed: {path}")
    atomic_json(path, value)


def split_identity(data, config, seed):
    parts = partition(data, seed, config)
    return parts, {
        "data_sha256": data.provenance["sha256"],
        "configuration": fingerprint(config),
        "train": parts[0].tolist(),
        "validation": parts[1].tolist(),
        "test": parts[2].tolist(),
    }
