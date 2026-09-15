"""Comparable-budget selection, persistent trials, and a one-time locked holdout."""

from __future__ import annotations

import importlib.metadata
import itertools
import json
import os
import threading
import time
import warnings

import joblib
import numpy as np
import psutil
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from qml_research.campaign import neural
from qml_research.campaign.runner import dump_model, metrics, threshold
from qml_research.campaign.storage import atomic_json, atomic_npz, check_stop, fingerprint
from qml_research.innovation.data import (
    OUTPUT,
    load,
    preprocess,
    seal,
    split_identity,
    transform,
)
from qml_research.innovation.models import GatedResidual, residual_inputs

BASES = ["logistic", "rbf_svm", "random_forest", "gradient_boosting", "mlp"]
VARIANTS = ["quantum", "ungated", "separable", "frozen", "classical", "no_quantum"]
FAMILIES = [f"{base}_{space}" for base in BASES for space in ["original", "reduced"]] + [
    "residual_" + variant for variant in VARIANTS
]


class Resources:
    def __enter__(self):
        self.process = psutil.Process()
        self.rss = self.private = 0
        self.available = float("inf")
        self.stop = threading.Event()

        def sample():
            while not self.stop.is_set():
                memory = self.process.memory_info()
                self.rss = max(self.rss, memory.rss)
                self.private = max(self.private, getattr(memory, "private", memory.rss))
                self.available = min(self.available, psutil.virtual_memory().available)
                self.stop.wait(0.1)

        self.worker = threading.Thread(target=sample, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.worker.join()

    def record(self):
        return {
            "peak_rss_gib": self.rss / 1024**3,
            "peak_private_gib": self.private / 1024**3,
            "minimum_free_gib": self.available / 1024**3,
        }


def specifications(family, config):
    count = config["trials_per_family"]
    base = family.rsplit("_", 1)[0]
    if base == "logistic":
        candidates = [
            {"C": c, "class_weight": w}
            for c, w in itertools.product([0.1, 1.0, 10.0, 0.01, 100.0, 1000.0], ["balanced", None])
        ]
    elif base == "rbf_svm":
        candidates = [
            {"C": c, "gamma": g}
            for c, g in itertools.product([0.1, 1.0, 10.0, 100.0], ["scale", 0.01, 0.1])
        ]
    elif base == "random_forest":
        candidates = [
            {"max_depth": d, "min_samples_leaf": leaf, "max_features": f}
            for d, leaf, f in itertools.product([None, 4, 8], [1, 3], ["sqrt", 0.8])
        ]
    elif base == "gradient_boosting":
        candidates = [
            {"learning_rate": lr, "max_leaf_nodes": leaves, "l2_regularization": l2}
            for lr, leaves, l2 in itertools.product([0.03, 0.1], [7, 15, 31], [0.0, 1.0])
        ]
    else:
        candidates = [
            {"layers": layer, "lr": lr, "weight_decay": decay}
            for layer, lr, decay in itertools.product([1, 2], [0.003, 0.01, 0.03], [0.0001, 0.01])
        ]
    # Fixed random ordering prevents tiny prefixes from varying only one weak parameter.
    order = np.random.default_rng(915).permutation(len(candidates))
    chosen = [candidates[i] for i in order[:count]]
    if len(chosen) != count:
        raise ValueError("Requested more than 12 distinct trials")
    if base == "mlp" or family.startswith("residual_"):
        for spec in chosen:
            spec.update(
                epochs=config["epochs"],
                patience=config["patience"],
                batch_size=config["batch_size"],
                hidden=16,
                dropout=0.0,
            )
    return chosen


def estimator(family, spec, seed):
    base = family.rsplit("_", 1)[0]
    if base == "logistic":
        return LogisticRegression(**spec, max_iter=3000, random_state=seed)
    if base == "rbf_svm":
        return SVC(**spec, class_weight="balanced", cache_size=128)
    if base == "random_forest":
        return RandomForestClassifier(
            **spec, n_estimators=200, class_weight="balanced", n_jobs=1, random_state=seed
        )
    if base == "gradient_boosting":
        return HistGradientBoostingClassifier(
            **spec, max_iter=200, early_stopping=False, random_state=seed
        )
    raise ValueError(family)


def is_neural(family):
    return family.startswith(("mlp_", "residual_"))


def model_for(family, features, spec, seed, config):
    if family.startswith("residual_"):
        return GatedResidual(config["qubits"], spec, seed, family.removeprefix("residual_"))
    return neural.make_model("classical_mlp", features, spec, seed)


def scores(model, x, family):
    if is_neural(family):
        return neural.scores(model, x)
    if hasattr(model, "decision_function"):
        return model.decision_function(x)
    return model.predict_proba(x)[:, 1]


def features_for(family, original, reduced, anchor):
    if family.startswith("residual_"):
        return residual_inputs(original, reduced, anchor)
    return reduced if family.endswith("_reduced") else original


def initialize(config):
    data = load()
    root = OUTPUT / config["name"]
    root.mkdir(parents=True, exist_ok=True)
    seal(root / "config.json", config)
    environment = {
        p: importlib.metadata.version(p)
        for p in ["torch", "pennylane", "numpy", "scikit-learn", "scipy"]
    }
    seal(root / "environment.json", environment)
    # Refuse silent checkpoint reuse after mathematical implementation changes.
    from pathlib import Path

    sources = [
        Path(__file__),
        Path(neural.__file__),
        Path(__file__).with_name("models.py"),
        Path(__file__).with_name("data.py"),
    ]
    seal(root / "implementation.json", {p.name: fingerprint(p.read_text()) for p in sources})
    atomic_json(
        root / "audit.json",
        {
            "source": data.provenance,
            "rows": len(data.y),
            "features": len(data.X.columns),
            "positive": int(data.y.sum()),
            "negative": int((data.y == 0).sum()),
            "duplicate_feature_rows": int(data.X.duplicated().sum()),
            "missing": {c: int(n) for c, n in data.X.isna().sum().items()},
            "endpoint": "CKD status, not future incidence; no patient IDs or chronology available",
        },
    )
    return data, root


def select(config):
    data, root = initialize(config)
    for seed in config["seeds"]:
        parts, identity = split_identity(data, config, seed)
        train, val, _ = parts
        folder = root / f"seed-{seed}"
        seal(folder / "split.json", identity)
        check_stop()
        prepared = folder / "preprocessing.joblib"
        if prepared.exists():
            saved = joblib.load(prepared)
            original, reduced = zip(
                *(transform(saved, data.X.iloc[p]) for p in [train, val]), strict=True
            )
        else:
            start = time.perf_counter()
            saved, original, reduced = preprocess(data, train, val, config["qubits"])
            anchor = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)
            anchor.fit(original[0], data.y[train])
            saved["anchor"] = anchor
            saved["preprocessing_and_anchor_seconds"] = time.perf_counter() - start
            dump_model(prepared, saved)
        anchor = saved["anchor"]
        for family in FAMILIES:
            destination = folder / family
            if (destination / "selected.json").exists():
                continue
            if config["allow_test"] and (OUTPUT / "holdout_unsealed.json").exists():
                raise ValueError("Test already opened; unfinished selection cannot continue")
            atomic_json(
                root / "status.json",
                {
                    "stage": "selection",
                    "seed": seed,
                    "family": family,
                    "pid": os.getpid(),
                    "test_evaluated": False,
                },
            )
            x, xv = [
                features_for(family, a, b, anchor) for a, b in zip(original, reduced, strict=True)
            ]
            trials = []
            for spec in specifications(family, config):
                check_stop()
                path = destination / "trials" / fingerprint(spec)
                path.mkdir(parents=True, exist_ok=True)
                result_path = path / "validation.json"
                if result_path.exists():
                    trials.append(json.loads(result_path.read_text()))
                    continue
                with Resources() as resources, warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    started = time.perf_counter()
                    tracking = {}
                    if is_neural(family):
                        model = model_for(family, x.shape[1], spec, seed, config)
                        tracking = neural.train(
                            model,
                            x,
                            data.y[train],
                            xv,
                            data.y[val],
                            spec=spec,
                            path=path,
                            seed=seed,
                        )
                    else:
                        model = estimator(family, spec, seed)
                        model.fit(x, data.y[train])
                        dump_model(path / "model.joblib", model)
                    prediction = scores(model, xv, family)
                    cut = threshold(data.y[val], prediction)
                    elapsed = time.perf_counter() - started
                record = {
                    "spec": spec,
                    "trial_id": path.name,
                    "validation": metrics(data.y[val], prediction, cut),
                    "threshold": cut,
                    "seconds": max(elapsed, tracking.get("elapsed", 0.0)),
                    "resources": resources.record(),
                    "warnings": sorted({str(w.message) for w in captured}),
                    "best_epoch": tracking.get("best_epoch"),
                    "steps": tracking.get("steps"),
                    "telemetry": model.telemetry() if isinstance(model, GatedResidual) else {},
                    "test_evaluated": False,
                }
                atomic_json(result_path, record)
                trials.append(record)
                print(
                    f"seed={seed} {family} val_auc={record['validation']['auroc']:.4f}", flush=True
                )
            best = max(trials, key=lambda r: r["validation"]["auroc"])
            atomic_json(
                destination / "selected.json",
                {
                    **best,
                    "total_tuning_seconds": sum(r["seconds"] for r in trials),
                    "peak_rss_gib": max(r["resources"]["peak_rss_gib"] for r in trials),
                    "trial_count": len(trials),
                },
            )
    atomic_json(root / "status.json", {"stage": "selection_complete", "test_evaluated": False})


def evaluate(config):
    if not config["allow_test"]:
        raise ValueError("Smoke configuration cannot access the test set")
    data, root = initialize(config)
    # All seeds and families must finish before ANY model gets a held-out prediction.
    selected = {}
    for seed in config["seeds"]:
        for family in FAMILIES:
            path = root / f"seed-{seed}" / family / "selected.json"
            if not path.exists():
                raise ValueError(f"Incomplete selection: {path}")
            selected[seed, family] = json.loads(path.read_text())
    seal(
        OUTPUT / "holdout_unsealed.json",
        {
            "experiment": config["name"],
            "config": fingerprint(config),
            "data_sha256": data.provenance["sha256"],
        },
    )
    records = []
    for seed in config["seeds"]:
        folder = root / f"seed-{seed}"
        saved = joblib.load(folder / "preprocessing.joblib")
        _, identity = split_identity(data, config, seed)
        seal(folder / "split.json", identity)
        test = np.array(identity["test"])
        original, reduced = transform(saved, data.X.iloc[test])
        reference = max(
            [f for f in FAMILIES if not f.startswith("residual_")],
            key=lambda f: selected[seed, f]["validation"]["auroc"],
        )
        for family in FAMILIES:
            destination = folder / family
            result_path = destination / "test.json"
            if result_path.exists():
                records.append(json.loads(result_path.read_text()))
                continue
            check_stop()
            best = selected[seed, family]
            x = features_for(family, original, reduced, saved["anchor"])
            trial = destination / "trials" / best["trial_id"]
            with Resources() as resources:
                start = time.perf_counter()
                if is_neural(family):
                    model = model_for(family, x.shape[1], best["spec"], seed, config)
                    model.load_state_dict(
                        torch.load(trial / "best.pt", weights_only=False, map_location="cpu")[
                            "model"
                        ]
                    )
                else:
                    model = joblib.load(trial / "model.joblib")
                prediction = scores(model, x, family)
                elapsed = time.perf_counter() - start
            atomic_npz(
                destination / "test_predictions.npz",
                indices=test,
                y=data.y[test],
                score=prediction,
                prediction=(prediction >= best["threshold"]).astype(int),
            )
            record = {
                "seed": seed,
                "family": family,
                "test": metrics(data.y[test], prediction, best["threshold"]),
                "selected": best,
                "reference": reference,
                "prediction_and_load_seconds": elapsed,
                "test_resources": resources.record(),
                "shared_preprocessing_anchor_seconds": saved["preprocessing_and_anchor_seconds"],
            }
            atomic_json(result_path, record)
            records.append(record)
    atomic_json(root / "results.json", records)
    atomic_json(root / "status.json", {"stage": "test_complete", "test_evaluated": True})
    return records
