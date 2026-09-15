"""Persistent model selection; test sets are used only after each family's search."""

from __future__ import annotations

import itertools
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qml_research.campaign import neural
from qml_research.campaign.data import Dataset, load_dataset, prepare, reduce_features, splits
from qml_research.campaign.kernels import fidelity, pauli_projection, states
from qml_research.campaign.storage import (
    OUTPUT,
    Paused,
    atomic_json,
    atomic_npz,
    check_stop,
    fingerprint,
    replace_file,
)


def dump_model(path: Path, model) -> None:
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(model, temporary)
    replace_file(temporary, path)


def threshold(y, score) -> float:
    fpr, tpr, thresholds = roc_curve(y, score)
    finite = np.isfinite(thresholds)
    return float(thresholds[finite][np.argmax((tpr - fpr)[finite])])


def metrics(y, score, cut) -> dict:
    prediction = score >= cut
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(np.mean(y == prediction)),
        "sensitivity": float(np.mean(prediction[y == 1])),
        "specificity": float(np.mean(~prediction[y == 0])),
    }


def trial_specs(family: str, config: dict) -> list[dict]:
    cs = config["C"]
    if family == "classical_logistic":
        return [
            {"C": c, "class_weight": w}
            for c, w in itertools.product(
                [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], [None, "balanced"]
            )
        ]
    if family == "classical_rbf_full":
        return [
            {"C": c, "gamma": g}
            for c, g in itertools.product(
                [0.01, 0.1, 1, 10, 100, 1000], [0.001, 0.01, 0.1, 1, 10, 100, "scale", "auto"]
            )
        ]
    if family == "classical_rbf_reduced":
        return [
            {"dimensions": d, "C": c, "gamma": g}
            for d, c, g in itertools.product([2, 4, 6, 8], cs, [0.1, 1.0, 10.0])
        ]
    if family == "classical_forest":
        return [
            {"depth": d, "leaf": leaf, "max_features": f}
            for d, leaf, f in itertools.product([None, 4, 8], [1, 5], ["sqrt", 0.75])
        ]
    if family == "classical_boosting":
        return [
            {"leaves": leaves, "l2": r, "learning_rate": lr}
            for leaves, r, lr in itertools.product([7, 15], [0.0, 1.0, 10.0], [0.05, 0.1])
        ]
    if family == "q_angle":
        return [
            {"dimensions": d, "scale": s, "kind": "angle", "repeats": 1, "C": c}
            for d, s, c in itertools.product([2, 4, 6, 8], [0.1, 0.3, 1.0], cs)
        ]
    if family == "q_iqp":
        return [
            {"dimensions": d, "scale": s, "kind": "iqp", "repeats": r, "C": c}
            for d, s, r, c in itertools.product([4, 6, 8], [0.1, 0.3], [1, 2], cs)
        ]
    if family == "q_projected":
        return [
            {"dimensions": d, "scale": s, "kind": "iqp", "repeats": 2, "gamma": g, "C": c}
            for d, s, g, c in itertools.product([4, 6, 8], [0.1, 0.3], [0.1, 1.0], cs)
        ]
    if family == "qcnn":
        return [
            {
                "dimensions": d,
                "lr": lr,
                "scale": scale,
                "epochs": config["image_qcnn_epochs"],
                "patience": config["neural_patience"],
                "batch_size": config["batch_size"],
                "weight_decay": 0.0001,
            }
            for d, lr, scale in itertools.product([4, 8], [0.003, 0.01], [0.3, 1.0])
        ]
    if family == "classical_cnn":
        return [
            {
                "lr": lr,
                "weight_decay": decay,
                "epochs": config["neural_epochs"],
                "patience": config["neural_patience"],
                "batch_size": config["batch_size"],
            }
            for lr, decay in itertools.product([0.001, 0.003], [0.0001, 0.001])
        ]
    return [
        {
            "dimensions": d,
            "lr": lr,
            "layers": layers,
            "epochs": config["neural_epochs"],
            "patience": config["neural_patience"],
            "batch_size": config["batch_size"],
            "weight_decay": 0.0001,
            "dropout": 0.0,
        }
        for d, lr, layers in itertools.product(
            config["neural_dimensions"], config["neural_learning_rates"], config["neural_layers"]
        )
    ]


class Fold:
    def __init__(self, dataset: Dataset, indices, seed: int, fold: int, config: dict):
        self.data, self.indices, self.seed, self.fold, self.config = (
            dataset,
            indices,
            seed,
            fold,
            config,
        )
        self.root = OUTPUT / config["name"] / dataset.name / f"seed-{seed}-fold-{fold}"
        self.root.mkdir(parents=True, exist_ok=True)
        identity = {
            "dataset": dataset.digest,
            "indices": [part.tolist() for part in indices],
            "config": config,
            "seed": seed,
            "fold": fold,
        }
        identity["fingerprint"] = fingerprint(identity)
        manifest = self.root / "split.json"
        if (
            manifest.exists()
            and json.loads(manifest.read_text())["fingerprint"] != identity["fingerprint"]
        ):
            raise ValueError("Existing fold belongs to different data or configuration")
        atomic_json(manifest, identity)
        pipeline, self.arrays = prepare(dataset, indices)
        dump_model(self.root / "preprocessing.joblib", pipeline)
        self.targets = tuple(dataset.y[part] for part in indices)
        self.reduced = {}
        self._last_key = None
        self._last_features = None

    def features(self, spec: dict, family: str, include_test=False):
        if "dimensions" not in spec:
            return self.arrays
        d = spec["dimensions"]
        if d not in self.reduced:
            reducer, values = reduce_features(self.arrays, d, self.seed)
            self.reduced[d] = values
            dump_model(self.root / f"pca-{d}.joblib", reducer)
        values = self.reduced[d]
        if family == "qcnn":
            return tuple(x * spec["scale"] for x in values)
        if not family.startswith("q_"):
            return values
        embedding_spec = {k: spec[k] for k in ["dimensions", "scale", "kind", "repeats"]}
        key = fingerprint({**embedding_spec, "projected": family == "q_projected"})
        if self._last_key == (key, include_test):
            return self._last_features
        circuit_states = []
        for part, x in zip(["train", "validation", "test"], values, strict=True):
            if part == "test" and not include_test:
                circuit_states.append(None)
                continue
            circuit_states.append(
                states(
                    x * spec["scale"],
                    kind=spec["kind"],
                    repeats=spec["repeats"],
                    path=self.root / "state_cache" / fingerprint(embedding_spec) / part,
                )
            )
        if family == "q_projected":
            projections = [pauli_projection(s) if s is not None else None for s in circuit_states]
            scaler = StandardScaler().fit(projections[0])
            result = tuple(scaler.transform(x) if x is not None else None for x in projections)
            if include_test:
                dump_model(self.root / family / "projection-scaler.joblib", scaler)
        else:
            result = tuple(
                fidelity(x, circuit_states[0]) if x is not None else None for x in circuit_states
            )
        self._last_key, self._last_features = (key, include_test), result
        return result

    def estimator(self, family: str, spec: dict, features: int):
        if family == "classical_logistic":
            return LogisticRegression(
                C=spec["C"],
                class_weight=spec["class_weight"],
                max_iter=3000,
                random_state=self.seed,
            )
        if family == "classical_forest":
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=spec["depth"],
                min_samples_leaf=spec["leaf"],
                max_features=spec["max_features"],
                class_weight="balanced",
                n_jobs=1,
                random_state=self.seed,
            )
        if family == "classical_boosting":
            return HistGradientBoostingClassifier(
                max_iter=200,
                max_leaf_nodes=spec["leaves"],
                l2_regularization=spec["l2"],
                learning_rate=spec["learning_rate"],
                random_state=self.seed,
            )
        if family in {"q_angle", "q_iqp"}:
            return SVC(kernel="precomputed", C=spec["C"], class_weight="balanced", cache_size=128)
        gamma = spec["gamma"]
        if isinstance(gamma, (int, float)):
            gamma /= features
        return SVC(C=spec["C"], gamma=gamma, class_weight="balanced", cache_size=128)

    @staticmethod
    def classical_scores(model, x):
        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(x))
        return np.asarray(model.predict_proba(x))[:, 1]

    def run_family(self, family: str):
        destination = self.root / family
        destination.mkdir(exist_ok=True)
        if (destination / "result.json").exists():
            return json.loads((destination / "result.json").read_text())
        neural_family = family not in self.config["kernel_families"]
        best_path = destination / "best.json"
        best = json.loads(best_path.read_text()) if best_path.exists() else None
        specs = trial_specs(family, self.config)
        for i, spec in enumerate(specs):
            check_stop()
            trial_path = destination / "trials" / fingerprint(spec)
            trial_path.mkdir(parents=True, exist_ok=True)
            record_path = trial_path / "validation.json"
            if record_path.exists():
                continue
            started = time.perf_counter()
            features = self.features(spec, family)
            model = None
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    if neural_family:
                        model = neural.make_model(family, features[0].shape[1], spec, self.seed)
                        tracking = neural.train(
                            model,
                            features[0],
                            self.targets[0],
                            features[1],
                            self.targets[1],
                            spec=spec,
                            path=trial_path,
                            seed=self.seed,
                        )
                        validation_score = neural.scores(model, features[1])
                    else:
                        model = self.estimator(family, spec, features[0].shape[1])
                        model.fit(features[0], self.targets[0])
                        tracking = {}
                        validation_score = self.classical_scores(model, features[1])
                cut = threshold(self.targets[1], validation_score)
                record = {
                    "spec": spec,
                    "validation": metrics(self.targets[1], validation_score, cut),
                    "threshold": cut,
                    "elapsed_seconds": time.perf_counter() - started,
                    "tracking": tracking,
                    "status": "completed",
                    "trial_id": trial_path.name,
                    "warnings": sorted({str(w.message) for w in captured}),
                    "test_set_evaluated": False,
                }
                if best is None or record["validation"]["auroc"] > best["validation"]["auroc"]:
                    if not neural_family:
                        dump_model(trial_path / "selected-model.joblib", model)
                    best = record
                    atomic_json(best_path, best)
                atomic_json(record_path, record)
                print(
                    f"{self.data.name} seed {self.seed} fold {self.fold} {family} "
                    f"trial {i + 1}/{len(specs)} val_AUROC={record['validation']['auroc']:.4f}",
                    flush=True,
                )
            except Paused:
                raise
            except Exception as exc:
                atomic_json(
                    trial_path / "failure.json",
                    {"spec": spec, "error": str(exc), "status": "failed"},
                )
                raise
        if best is None:
            raise RuntimeError("No successful tuning trial")
        check_stop()
        features = self.features(best["spec"], family, include_test=True)
        if neural_family:
            model = neural.make_model(family, features[0].shape[1], best["spec"], self.seed)
            selected = destination / "trials" / best["trial_id"] / "best.pt"
            saved = torch.load(selected, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"])
            prediction = neural.scores(model, features[2])
        else:
            model = joblib.load(destination / "trials" / best["trial_id"] / "selected-model.joblib")
            prediction = self.classical_scores(model, features[2])
        result = {
            "dataset": self.data.name,
            "seed": self.seed,
            "fold": self.fold,
            "family": family,
            "status": "completed",
            "selected_spec": best["spec"],
            "validation": best["validation"],
            "threshold": best["threshold"],
            "test": metrics(self.targets[2], prediction, best["threshold"]),
            "train_rows": len(self.targets[0]),
            "validation_rows": len(self.targets[1]),
            "test_rows": len(self.targets[2]),
            "tuning_trials": len(specs),
            "data_hash": self.data.digest,
            "selection": "validation AUROC only; first trial wins ties",
            "kind": "quantum"
            if family.startswith("q_") or family in {"hqmlp", "reupload_vqc", "qcnn"}
            else "classical",
            "training_tracking": best.get("tracking", {}),
        }
        atomic_npz(
            destination / "test_predictions.npz",
            indices=self.indices[2],
            y=self.targets[2],
            score=prediction,
            prediction=(prediction >= best["threshold"]).astype(int),
        )
        atomic_json(destination / "result.json", result)
        return result


def run(config: dict, *, stage: str, datasets=None, seeds=None, fold_limit=None):
    from qml_research.campaign.report import report

    root = OUTPUT / config["name"]
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise ValueError("Campaign configuration changed; use a new campaign name")
    atomic_json(manifest, config)
    base_families = config["kernel_families"] if stage == "kernels" else config["neural_families"]
    try:
        for dataset_name in datasets or config["datasets"]:
            data = load_dataset(dataset_name)
            families = list(base_families)
            if dataset_name == "breastmnist" and stage == "neural":
                families += config["image_neural_families"]
            for seed in seeds or config["seeds"]:
                for fold, indices in splits(data, seed, config["outer_folds"]):
                    if fold_limit is not None and fold >= fold_limit:
                        break
                    check_stop()
                    current = Fold(data, indices, seed, fold, config)
                    for family in families:
                        atomic_json(
                            root / "status.json",
                            {
                                "status": "running",
                                "stage": stage,
                                "dataset": dataset_name,
                                "seed": seed,
                                "fold": fold,
                                "family": family,
                                "pid": os.getpid(),
                                "updated_at": time.time(),
                            },
                        )
                        current.run_family(family)
                        report(config)
                    del current
        atomic_json(
            root / "status.json",
            {
                "status": "completed_requested_scope",
                "stage": stage,
                "datasets": datasets or config["datasets"],
                "seeds": seeds or config["seeds"],
                "fold_limit": fold_limit,
            },
        )
    except (Paused, KeyboardInterrupt):
        atomic_json(
            root / "status.json",
            {
                "status": "paused",
                "stage": stage,
                "resume": "Rerun the same command; completed trials and batch states are reused",
            },
        )
        report(config)
        return 2
    except Exception as exc:
        atomic_json(root / "status.json", {"status": "failed", "stage": stage, "error": str(exc)})
        report(config)
        raise
    report(config)
    return 0
