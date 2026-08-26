"""End-to-end execution, tuning, noise evaluation, inference, and reports."""

from __future__ import annotations

import itertools
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image

from qml_research.qcnn.config import PROJECT_ROOT, QCNNConfig, qcnn_config_from_checkpoint
from qml_research.qcnn.data import DatasetSplits, load_splits, prepare_features, write_manifest
from qml_research.qcnn.evaluation import (
    choose_threshold,
    evaluate_scores,
    stratified_bootstrap_interval,
)
from qml_research.qcnn.models import (
    PretrainedWeightsUnavailable,
    build_image_model,
    classical_scores,
    make_reduced_classical_model,
    total_parameter_count,
    trainable_parameter_count,
)
from qml_research.qcnn.quantum import HierarchicalQCNN
from qml_research.qcnn.tracking import RunDirectory, stable_hash, system_metadata, write_json
from qml_research.qcnn.training import (
    predict_image_model,
    predict_qcnn,
    train_image_model,
    train_qcnn,
)
from qml_research.qcnn.visualization import (
    load_records,
    plot_architecture,
    plot_dataset,
    plot_evaluation,
    plot_history,
    plot_pca,
    plot_summary,
)

REDUCED_MODELS = {"logistic", "linear_svm", "rbf_svm", "random_forest", "reduced_mlp"}
IMAGE_MODELS = {
    "small_cnn",
    "resnet18_random",
    "resnet18_imagenet",
    "mobilenet_v3_small_imagenet",
}


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _check_expensive(config: QCNNConfig, confirmed: bool) -> dict[str, Any]:
    estimate = config.estimated_resources()
    if estimate["expensive"] and not (confirmed or config.confirm_expensive):
        reasons = ", ".join(estimate["expensive_reasons"])
        raise RuntimeError(
            f"expensive run blocked ({reasons}); inspect the estimate and rerun with "
            "--confirm-expensive"
        )
    return estimate


def _identity(
    config: QCNNConfig,
    splits: DatasetSplits,
    *,
    seed: int,
    fold: int,
    model: str,
) -> dict[str, Any]:
    return {
        "config": config.to_dict(),
        "dataset_provenance": splits.provenance,
        "split_indices": {
            name: getattr(splits, name).indices.tolist() for name in ("train", "validation", "test")
        },
        "seed": seed,
        "fold": fold,
        "model": model,
    }


def _predictions_frame(
    splits: DatasetSplits,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, split, scores in (
        ("validation", splits.validation, validation_scores),
        ("test", splits.test, test_scores),
    ):
        for index, target, score in zip(split.indices, split.y, scores, strict=True):
            rows.append(
                {
                    "split": split_name,
                    "dataset_index": int(index),
                    "y_true": int(target),
                    "malignant_score": float(score),
                    "prediction": int(score >= threshold),
                    "threshold": float(threshold),
                }
            )
    return pd.DataFrame(rows)


def _metrics_with_uncertainty(
    splits: DatasetSplits,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    config: QCNNConfig,
    *,
    training_seconds: float,
    prediction_seconds: float,
    seed: int,
) -> tuple[float, dict[str, Any]]:
    threshold = choose_threshold(splits.validation.y, validation_scores, config.threshold_strategy)
    metrics = evaluate_scores(
        splits.test.y,
        test_scores,
        threshold=threshold,
        training_time_seconds=training_seconds,
        prediction_time_seconds=prediction_seconds,
    )
    metrics["default_0.5"] = evaluate_scores(
        splits.test.y,
        test_scores,
        threshold=0.5,
        training_time_seconds=training_seconds,
        prediction_time_seconds=prediction_seconds,
    )
    metrics["confidence_intervals"] = {
        metric: stratified_bootstrap_interval(
            splits.test.y,
            test_scores,
            metric=metric,
            threshold=threshold,
            iterations=config.bootstrap_iterations,
            seed=seed,
        )
        for metric in ("auroc", "balanced_accuracy", "sensitivity", "specificity")
    }
    return threshold, metrics


def _write_completed_run(
    run: RunDirectory,
    *,
    record: dict[str, Any],
    splits: DatasetSplits,
    validation_scores: np.ndarray,
    test_scores: np.ndarray,
    threshold: float,
    history: list[dict[str, Any]],
) -> None:
    frame = _predictions_frame(splits, validation_scores, test_scores, threshold)
    frame.to_csv(run.path / "predictions.csv", index=False)
    write_manifest(splits, run.path / "split_manifest.json")
    run.write("history.json", {"history": history})
    run.write("resolved_config.json", record["config"])
    run.write("record.json", record)
    plot_history(history, run.path / "figures")
    plot_evaluation(splits.test.y, test_scores, record["metrics"], run.path / "figures")


def _base_record(
    config: QCNNConfig,
    splits: DatasetSplits,
    seed: int,
    fold: int,
    model: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "running",
        "experiment": config.experiment_name,
        "profile": config.profile,
        "dataset": config.dataset,
        "model": model,
        "seed": seed,
        "fold": fold,
        "config": config.to_dict(),
        "dataset_manifest": splits.manifest(),
        "system": metadata,
        "scientific_boundary": (
            "Research-only simulated benchmark; not clinically validated and not evidence "
            "of computational quantum advantage."
        ),
    }


def _record_failure(
    run: RunDirectory,
    config: QCNNConfig,
    splits: DatasetSplits,
    seed: int,
    fold: int,
    model: str,
    metadata: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    record = _base_record(config, splits, seed, fold, model, metadata)
    record.update(
        {
            "status": "failed",
            "run_id": run.run_id,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
    )
    run.write("record.json", record)
    return record


def _run_qcnn(
    run: RunDirectory,
    config: QCNNConfig,
    splits: DatasetSplits,
    features: dict[str, np.ndarray],
    reducer_metadata: dict[str, Any],
    seed: int,
    fold: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    record = _base_record(config, splits, seed, fold, "qcnn", metadata)
    run.write("record.json", record)
    result = train_qcnn(
        features["train"],
        splits.train.y,
        features["validation"],
        splits.validation.y,
        config,
        seed,
    )
    prediction_started = time.perf_counter()
    validation_scores = predict_qcnn(result.model, features["validation"])
    test_scores = predict_qcnn(result.model, features["test"])
    prediction_seconds = time.perf_counter() - prediction_started
    threshold, metrics = _metrics_with_uncertainty(
        splits,
        validation_scores,
        test_scores,
        config,
        training_seconds=result.training_time_seconds,
        prediction_seconds=prediction_seconds,
        seed=seed,
    )
    resources = result.model.resource_summary().to_dict()
    resources["circuit_executions"] = result.model.circuit_executions
    resources["shots"] = config.shots
    resources["backend"] = config.backend
    resources["gradient_method"] = config.gradient_method
    checkpoint = run.path / "checkpoint.pt"
    torch.save(
        {
            "model_type": "qcnn",
            "model_state": result.model.state_dict(),
            "config": config.to_dict(),
            "threshold": threshold,
            "run_id": run.run_id,
            "reducer": reducer_metadata,
        },
        checkpoint,
    )
    (run.path / "circuit.txt").write_text(result.model.draw_text(), encoding="utf-8")
    plot_architecture(config.qubits, run.path / "figures")
    plot_pca(reducer_metadata.get("explained_variance_ratio"), run.path / "figures")
    record.update(
        {
            "status": "completed",
            "run_id": run.run_id,
            "metrics": metrics,
            "resources": resources,
            "preprocessing": reducer_metadata,
            "best_epoch": result.best_epoch,
            "stopping_reason": result.stopping_reason,
            "peak_memory_bytes": result.peak_memory_bytes,
            "checkpoint": str(checkpoint),
        }
    )
    _write_completed_run(
        run,
        record=record,
        splits=splits,
        validation_scores=validation_scores,
        test_scores=test_scores,
        threshold=threshold,
        history=result.history,
    )
    return record


def _run_reduced_model(
    run: RunDirectory,
    name: str,
    config: QCNNConfig,
    splits: DatasetSplits,
    features: dict[str, np.ndarray],
    reducer_metadata: dict[str, Any],
    seed: int,
    fold: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    record = _base_record(config, splits, seed, fold, name, metadata)
    run.write("record.json", record)
    model = make_reduced_classical_model(name, seed)
    started = time.perf_counter()
    model.fit(features["train"], splits.train.y)
    training_seconds = time.perf_counter() - started
    prediction_started = time.perf_counter()
    validation_scores = classical_scores(model, features["validation"])
    test_scores = classical_scores(model, features["test"])
    prediction_seconds = time.perf_counter() - prediction_started
    threshold, metrics = _metrics_with_uncertainty(
        splits,
        validation_scores,
        test_scores,
        config,
        training_seconds=training_seconds,
        prediction_seconds=prediction_seconds,
        seed=seed,
    )
    record.update(
        {
            "status": "completed",
            "run_id": run.run_id,
            "metrics": metrics,
            "resources": {"trainable_parameters": None, "backend": "scikit-learn"},
            "preprocessing": reducer_metadata,
            "best_epoch": None,
            "stopping_reason": "fit_completed",
            "peak_memory_bytes": None,
            "checkpoint": None,
        }
    )
    _write_completed_run(
        run,
        record=record,
        splits=splits,
        validation_scores=validation_scores,
        test_scores=test_scores,
        threshold=threshold,
        history=[],
    )
    return record


def _run_image_model(
    run: RunDirectory,
    name: str,
    config: QCNNConfig,
    splits: DatasetSplits,
    seed: int,
    fold: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    record = _base_record(config, splits, seed, fold, name, metadata)
    run.write("record.json", record)
    try:
        model, representation = build_image_model(
            name,
            pretrained_download=config.pretrained_download,
            pretrained_mode=config.pretrained_mode,
        )
    except PretrainedWeightsUnavailable as exc:
        record.update({"status": "skipped", "skip_reason": str(exc), "run_id": run.run_id})
        run.write("record.json", record)
        return record
    result = train_image_model(
        model,
        representation,
        splits.train.X,
        splits.train.y,
        splits.validation.X,
        splits.validation.y,
        config,
        seed,
    )
    device = next(result.model.parameters()).device
    prediction_started = time.perf_counter()
    validation_scores = predict_image_model(
        result.model,
        splits.validation.X,
        representation,
        batch_size=config.image_batch_size,
        device=device,
    )
    test_scores = predict_image_model(
        result.model,
        splits.test.X,
        representation,
        batch_size=config.image_batch_size,
        device=device,
    )
    prediction_seconds = time.perf_counter() - prediction_started
    threshold, metrics = _metrics_with_uncertainty(
        splits,
        validation_scores,
        test_scores,
        config,
        training_seconds=result.training_time_seconds,
        prediction_seconds=prediction_seconds,
        seed=seed,
    )
    checkpoint = run.path / "checkpoint.pt"
    torch.save(
        {
            "model_type": name,
            "model_state": result.model.cpu().state_dict(),
            "config": config.to_dict(),
            "threshold": threshold,
            "run_id": run.run_id,
            "representation": representation,
        },
        checkpoint,
    )
    record.update(
        {
            "status": "completed",
            "run_id": run.run_id,
            "metrics": metrics,
            "resources": {
                "trainable_parameters": trainable_parameter_count(result.model),
                "total_parameters": total_parameter_count(result.model),
                "backend": str(device),
            },
            "preprocessing": {"representation": representation},
            "best_epoch": result.best_epoch,
            "stopping_reason": result.stopping_reason,
            "peak_memory_bytes": result.peak_memory_bytes,
            "checkpoint": str(checkpoint),
        }
    )
    _write_completed_run(
        run,
        record=record,
        splits=splits,
        validation_scores=validation_scores,
        test_scores=test_scores,
        threshold=threshold,
        history=result.history,
    )
    return record


def run_experiment(
    config: QCNNConfig,
    *,
    download: bool = False,
    force: bool = False,
    confirm_expensive: bool = False,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    estimate = _check_expensive(config, confirm_expensive)
    if checkpoint is not None and not config.noise_enabled:
        raise ValueError("--checkpoint is only valid with a noisy-simulation profile")
    if config.backend == "hardware_export":
        return export_hardware_bundle(config)
    if config.noise_enabled:
        return evaluate_noise(config, checkpoint)
    output_root = _resolve(config.output_dir)
    metadata = system_metadata(PROJECT_ROOT)
    records: list[dict[str, Any]] = []
    folds = range(config.cv_folds) if config.dataset == "wdbc" else range(1)
    dataset_plotted = False
    for seed in config.seeds:
        for fold in folds:
            splits = load_splits(config, download=download, seed=seed, fold=fold)
            features, reducer = prepare_features(splits, config, seed)
            reducer_metadata = reducer.metadata()
            if not dataset_plotted:
                plot_dataset(
                    splits.train.X if splits.is_image else None,
                    splits.train.y,
                    output_root / "figures" / config.profile,
                )
                dataset_plotted = True
            for model_name in config.models:
                identity = _identity(config, splits, seed=seed, fold=fold, model=model_name)
                run = RunDirectory(output_root, identity)
                if config.cache_enabled and run.completed() and not force:
                    records.append(json.loads(run.record_path.read_text(encoding="utf-8")))
                    continue
                try:
                    if model_name == "qcnn":
                        record = _run_qcnn(
                            run,
                            config,
                            splits,
                            features,
                            reducer_metadata,
                            seed,
                            fold,
                            metadata,
                        )
                    elif model_name in REDUCED_MODELS:
                        record = _run_reduced_model(
                            run,
                            model_name,
                            config,
                            splits,
                            features,
                            reducer_metadata,
                            seed,
                            fold,
                            metadata,
                        )
                    elif model_name in IMAGE_MODELS:
                        if not splits.is_image:
                            record = _base_record(config, splits, seed, fold, model_name, metadata)
                            record.update(
                                {
                                    "status": "skipped",
                                    "run_id": run.run_id,
                                    "skip_reason": (
                                        "image model is not applicable to WDBC tabular data"
                                    ),
                                }
                            )
                            run.write("record.json", record)
                        else:
                            record = _run_image_model(
                                run, model_name, config, splits, seed, fold, metadata
                            )
                    else:
                        raise AssertionError(model_name)  # pragma: no cover
                except Exception as error:
                    record = _record_failure(
                        run,
                        config,
                        splits,
                        seed,
                        fold,
                        model_name,
                        metadata,
                        error,
                    )
                records.append(record)
    report = generate_report(output_root)
    return {
        "profile": config.profile,
        "dataset": config.dataset,
        "estimate": estimate,
        "completed": sum(record.get("status") == "completed" for record in records),
        "skipped": sum(record.get("status") == "skipped" for record in records),
        "failed": sum(record.get("status") == "failed" for record in records),
        "run_ids": [record.get("run_id") for record in records],
        "report": report,
    }


def verify_research() -> dict[str, Any]:
    required = {
        "literature_review": PROJECT_ROOT / "research" / "qcnn-literature-review.md",
        "architecture_decision": PROJECT_ROOT / "research" / "qcnn-architecture-decision.md",
        "qrnn_feasibility": PROJECT_ROOT / "research" / "qrnn-feasibility.md",
        "implementation_record": PROJECT_ROOT / "specs" / "qcnn-breast-cancer.md",
    }
    paper_markers = (
        "105560",
        "2509.14277",
        "L042060",
        "2604.26110",
        "Exponential concentration in quantum kernel methods",
        "1810.03787",
        "2403.07059",
    )
    missing_files = [name for name, path in required.items() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"required research records are missing: {missing_files}")
    review = required["literature_review"].read_text(encoding="utf-8")
    missing = [marker for marker in paper_markers if marker not in review]
    if missing:
        raise ValueError(f"literature review is missing source markers: {missing}")
    return {
        "status": "verified",
        "documents": {name: str(path) for name, path in required.items() if path.exists()},
        "source_markers": list(paper_markers),
    }


def prepare_or_verify_data(config: QCNNConfig, *, download: bool) -> dict[str, Any]:
    splits = load_splits(config, download=download)
    manifest = splits.manifest()
    path = _resolve(config.output_dir) / "data" / f"{config.dataset}-{config.profile}.json"
    write_json(path, manifest)
    return {"status": "verified", "manifest": str(path), **manifest}


def _find_latest_qcnn_checkpoint(output_root: Path) -> Path:
    candidates = sorted(
        output_root.glob("runs/*/checkpoint.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        payload = torch.load(candidate, map_location="cpu", weights_only=False)
        if payload.get("model_type") == "qcnn":
            return candidate
    raise FileNotFoundError("no completed QCNN checkpoint is available for noise evaluation")


def evaluate_noise(config: QCNNConfig, checkpoint: str | Path | None = None) -> dict[str, Any]:
    output_root = _resolve(config.output_dir)
    checkpoint_path = Path(checkpoint) if checkpoint else _find_latest_qcnn_checkpoint(output_root)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_config = qcnn_config_from_checkpoint(payload["config"])
    splits = load_splits(source_config, download=False)
    features, _ = prepare_features(splits, source_config, source_config.seeds[0])
    rows: list[dict[str, Any]] = []
    combinations = list(itertools.product(config.shot_values, config.noise_probabilities))
    for shots, probability in combinations:
        model = HierarchicalQCNN(
            source_config.qubits,
            backend="default.mixed",
            shots=shots,
            gradient_method="parameter-shift",
            noise_probability=probability,
            seed=source_config.seeds[0],
        )
        model.load_state_dict(payload["model_state"])
        started = time.perf_counter()
        scores = predict_qcnn(model, features["test"])
        elapsed = time.perf_counter() - started
        metrics = evaluate_scores(
            splits.test.y,
            scores,
            threshold=float(payload["threshold"]),
            training_time_seconds=0.0,
            prediction_time_seconds=elapsed,
        )
        rows.append(
            {
                "shots": shots,
                "noise_probability": probability,
                "metrics": metrics,
                "circuit_executions": model.circuit_executions,
            }
        )
    result = {
        "status": "completed",
        "checkpoint": str(checkpoint_path),
        "evaluations": rows,
        "scientific_boundary": "simulated depolarizing and finite-shot evaluation; not hardware",
    }
    write_json(output_root / "noise" / f"{stable_hash(result)}.json", result)
    return result


def export_hardware_bundle(config: QCNNConfig) -> dict[str, Any]:
    if config.real_hardware_submit:
        raise ValueError("hardware submission is not permitted")
    model = HierarchicalQCNN(
        config.qubits,
        backend="default.qubit",
        shots=config.shots,
        gradient_method=config.gradient_method,
        seed=config.seeds[0],
    )
    output_root = _resolve(config.output_dir) / "hardware_export"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "circuit.txt").write_text(model.draw_text(), encoding="utf-8")
    payload = {
        "status": "ready_not_submitted",
        "provider": config.real_hardware_provider,
        "shots": config.shots,
        "resources": model.resource_summary().to_dict(),
        "requirements": [
            "select provider/backend explicitly",
            "transpile to backend basis/connectivity",
            "recheck depth and two-qubit gates",
            "obtain separate authorization before submission",
        ],
    }
    write_json(output_root / "bundle.json", payload)
    return payload


def tune_qcnn(config: QCNNConfig, *, confirm_expensive: bool = False) -> dict[str, Any]:
    if "qcnn" not in config.models:
        config = replace(config, models=["qcnn"])
    _check_expensive(config, confirm_expensive)
    rng = random.Random(config.seeds[0])
    keys = sorted(config.tuning_space)
    all_options = list(itertools.product(*(config.tuning_space[key] for key in keys)))
    rng.shuffle(all_options)
    trials = all_options[: config.tuning_trials]
    splits = load_splits(config, download=False)
    features, reducer = prepare_features(splits, config, config.seeds[0])
    records: list[dict[str, Any]] = []
    for trial_index, values in enumerate(trials):
        parameters = dict(zip(keys, values, strict=True))
        trial_config = replace(config, **parameters, models=["qcnn"], seeds=[config.seeds[0]])
        result = train_qcnn(
            features["train"],
            splits.train.y,
            features["validation"],
            splits.validation.y,
            trial_config,
            config.seeds[0],
        )
        validation_scores = predict_qcnn(result.model, features["validation"])
        metrics = evaluate_scores(
            splits.validation.y,
            validation_scores,
            threshold=0.5,
            training_time_seconds=result.training_time_seconds,
            prediction_time_seconds=0.0,
        )
        records.append(
            {
                "trial": trial_index,
                "parameters": parameters,
                "validation_auroc": metrics["auroc"],
                "validation_balanced_accuracy": metrics["balanced_accuracy"],
                "best_epoch": result.best_epoch,
            }
        )
    ranked = sorted(records, key=lambda row: row["validation_auroc"], reverse=True)
    payload = {
        "status": "completed",
        "test_set_evaluated": False,
        "reducer": reducer.metadata(),
        "trials": ranked,
        "selected": ranked[0] if ranked else None,
    }
    output = _resolve(config.output_dir) / "tuning" / f"{config.fingerprint()}.json"
    write_json(output, payload)
    return {**payload, "output": str(output)}


def generate_report(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    records = load_records(root)
    rows: list[dict[str, Any]] = []
    for record in records:
        metrics = record.get("metrics", {})
        rows.append(
            {
                "run_id": record.get("run_id"),
                "status": record.get("status"),
                "profile": record.get("profile"),
                "dataset": record.get("dataset"),
                "model": record.get("model"),
                "seed": record.get("seed"),
                "fold": record.get("fold"),
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "sensitivity": metrics.get("sensitivity"),
                "specificity": metrics.get("specificity"),
                "f1": metrics.get("f1"),
                "auroc": metrics.get("auroc"),
                "auprc": metrics.get("auprc"),
                "mcc": metrics.get("mcc"),
                "false_positives": metrics.get("false_positives"),
                "false_negatives": metrics.get("false_negatives"),
                "training_time_seconds": metrics.get("training_time_seconds"),
                "peak_memory_bytes": record.get("peak_memory_bytes"),
            }
        )
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(reports / "normalized_results.csv", index=False)
    completed = frame[frame["status"] == "completed"] if not frame.empty else frame
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "auroc",
        "auprc",
        "mcc",
        "training_time_seconds",
    ]
    summary = (
        completed.groupby(["dataset", "model"])[metric_columns].agg(["mean", "std", "count"])
        if not completed.empty
        else pd.DataFrame()
    )
    summary.to_csv(reports / "summary.csv")
    figures = plot_summary(records, root / "figures" / "summary")
    return {
        "records": len(records),
        "completed": int(len(completed)),
        "normalized_results": str(reports / "normalized_results.csv"),
        "summary": str(reports / "summary.csv"),
        "figures": [str(path) for path in figures],
    }


def inspect_run(output_root: str | Path, run_id: str) -> dict[str, Any]:
    path = Path(output_root) / "runs" / run_id / "record.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def predict_image(checkpoint: str | Path, image_path: str | Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = qcnn_config_from_checkpoint(payload["config"])
    if payload.get("model_type") != "qcnn":
        raise ValueError("predict currently accepts a QCNN checkpoint")
    if config.reducer != "spatial_pool":
        raise ValueError("standalone image prediction requires a spatial_pool checkpoint")
    image = Image.open(image_path).convert("L").resize((28, 28))
    array = np.asarray(image, dtype=np.uint8)[None, ...]
    from qml_research.qcnn.data import FeatureReducer

    reducer = FeatureReducer(
        "spatial_pool", config.reduced_features, config.seeds[0], is_image=True
    )
    features = reducer.fit_transform(array)
    model = HierarchicalQCNN(
        config.qubits,
        backend="lightning.qubit",
        shots=None,
        gradient_method="adjoint",
        seed=config.seeds[0],
    )
    model.load_state_dict(payload["model_state"])
    probability = float(predict_qcnn(model, features)[0])
    threshold = float(payload["threshold"])
    return {
        "run_id": payload["run_id"],
        "image": str(image_path),
        "malignant_probability": probability,
        "threshold": threshold,
        "prediction": "malignant" if probability >= threshold else "normal_or_benign",
        "warning": "Research-only output; not a diagnosis or clinically validated device.",
    }
