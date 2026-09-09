"""End-to-end bounded UCI smoke runner with protected evidence artifacts."""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
import psutil
import sklearn
import torch
import yaml
from sklearn.metrics import average_precision_score
from sklearn.svm import SVC

from qml_research.ehr_ihd.config import IHDConfig
from qml_research.ehr_ihd.data import (
    UCI_FEATURES,
    dataset_report,
    deterministic_split,
    load_uci_cleveland,
)
from qml_research.ehr_ihd.evaluation import (
    binary_metrics,
    bootstrap_intervals,
    calibrated_probabilities,
    fit_platt_calibrator,
    permutation_explanation,
    perturbation_explanation,
    select_threshold,
)
from qml_research.ehr_ihd.features import FittedFeaturePipeline, consensus_feature_ranking
from qml_research.ehr_ihd.models import (
    HybridQMLP,
    PauliStatevectorKernel,
    classical_model,
    decision_scores,
    qmlp_scores,
    train_qmlp,
)
from qml_research.ehr_ihd.predict import predict_record
from qml_research.results import git_state


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True), encoding="utf-8")


def _run_directory(config: IHDConfig, run_id: str | None) -> tuple[str, Path]:
    if run_id is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"uci-smoke-{stamp}-{config.fingerprint()}"
    root = config.output_dir / run_id
    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {root}")
    for name in (
        "preprocessing",
        "models",
        "predictions",
        "metrics",
        "calibration",
        "plots",
        "explanations",
        "scaling",
        "logs",
    ):
        (root / name).mkdir(parents=True, exist_ok=False)
    return run_id, root


def _environment() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage(str(Path.cwd()))
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "cpu_logical": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "memory_total_bytes": int(memory.total),
        "memory_available_at_start_bytes": int(memory.available),
        "disk_free_bytes": int(disk.free),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "pennylane": qml.__version__,
        "torch": torch.__version__,
        "qiskit_machine_learning": "not_installed",
        "git": git_state(Path.cwd()),
    }


def _balanced_subset(indices: np.ndarray, target: np.ndarray, limit: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    per_class = max(2, limit // 2)
    selected = []
    for label in (0, 1):
        candidates = indices[target[indices] == label]
        selected.extend(rng.choice(candidates, size=min(per_class, len(candidates)), replace=False))
    return np.sort(np.asarray(selected, dtype=int))


def _evaluate(
    *,
    model_name: str,
    validation_target: np.ndarray,
    validation_score: np.ndarray,
    test_target: np.ndarray,
    test_score: np.ndarray,
    training_time: float,
    inference_time: float,
    peak_memory_mb: float,
    threshold_strategy: str,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[dict[str, Any], Any, float, np.ndarray]:
    calibrator = fit_platt_calibrator(validation_score, validation_target)
    validation_probability = calibrated_probabilities(calibrator, validation_score)
    threshold = select_threshold(validation_target, validation_probability, threshold_strategy)
    test_probability = calibrated_probabilities(calibrator, test_score)
    metrics = binary_metrics(
        test_target,
        test_probability,
        threshold,
        training_time_seconds=training_time,
        inference_time_seconds=inference_time,
        peak_memory_mb=peak_memory_mb,
    )
    metrics["model"] = model_name
    metrics["calibration"] = "platt_validation_only"
    metrics["confidence_intervals"] = bootstrap_intervals(
        test_target, test_probability, iterations=bootstrap_iterations, seed=seed
    )
    return metrics, calibrator, threshold, test_probability


def _save_predictions(
    path: Path,
    patient_ids: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> None:
    pd.DataFrame(
        {
            "patient_id": patient_ids,
            "target": target,
            "uncalibrated_score": score,
            "calibrated_probability": probability,
            "prediction": (probability >= threshold).astype(int),
        }
    ).to_csv(path, index=False)


def _plot_metrics(records: list[dict[str, Any]], path: Path) -> None:
    names = [record["model"] for record in records]
    auroc = [record["auroc"] for record in records]
    auprc = [record["auprc"] for record in records]
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(max(8, len(names) * 1.2), 4.8))
    axis.bar(positions - 0.18, auroc, width=0.36, label="AUROC")
    axis.bar(positions + 0.18, auprc, width=0.36, label="AUPRC")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Held-out test metric")
    axis.set_xticks(positions, names, rotation=30, ha="right")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_uci_smoke(config: IHDConfig, *, run_id: str | None = None) -> dict[str, Any]:
    if config.track not in {"uci_diagnostic", "paper_replication"}:
        raise ValueError("run_uci_smoke requires the UCI diagnostic or paper-replication track")
    run_id, root = _run_directory(config, run_id)
    started = time.perf_counter()
    process = psutil.Process()
    environment = _environment()
    _write_json(root / "environment.json", environment)
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config.payload, sort_keys=False), encoding="utf-8"
    )

    dataset = load_uci_cleveland()
    split = deterministic_split(dataset, config.seed)
    _write_json(root / "dataset_metadata.json", dataset.metadata)
    _write_json(root / "cohort_report.json", dataset_report(dataset, split))
    _write_json(root / "censoring_report.json", {"applicable": False, "reason": "diagnostic track"})
    _write_json(
        root / "split_manifest.json",
        {
            name: {
                "patient_ids": dataset.patient_ids[indices],
                "target": dataset.target[indices],
            }
            for name, indices in split.items()
        },
    )
    _write_json(
        root / "feature_dictionary.json",
        {
            name: {"source_field": name, "input_order": index}
            for index, name in enumerate(UCI_FEATURES)
        },
    )

    feature_count = int(config.section("quantum").get("features", 4))
    selected, rankings = consensus_feature_ranking(
        dataset.frame.iloc[split["train"]],
        dataset.target[split["train"]],
        n_features=feature_count,
        seed=config.seed,
    )
    _write_json(root / "feature_rankings.json", rankings)
    _write_json(
        root / "selected_features.json",
        {"features": selected, "count": len(selected), "selection_fit": "training_only"},
    )
    pipeline = FittedFeaturePipeline.fit(dataset.frame.iloc[split["train"]], selected)
    joblib.dump(pipeline, root / "preprocessing" / "selected_pipeline.joblib")

    train_values = pipeline.transform(dataset.frame.iloc[split["train"]])
    validation_values = pipeline.transform(dataset.frame.iloc[split["validation"]])
    test_values = pipeline.transform(dataset.frame.iloc[split["test"]])
    train_quantum = pipeline.transform(dataset.frame.iloc[split["train"]], quantum=True)
    validation_quantum = pipeline.transform(dataset.frame.iloc[split["validation"]], quantum=True)
    test_quantum = pipeline.transform(dataset.frame.iloc[split["test"]], quantum=True)
    threshold_strategy = str(config.section("evaluation").get("threshold_strategy", "youden_j"))
    bootstrap_iterations = int(config.section("training").get("bootstrap_iterations", 200))
    metrics_records: list[dict[str, Any]] = []

    model_names = list(
        config.section("models").get(
            "classical", ["logistic", "linear_svm", "rbf_svm", "random_forest", "classical_mlp"]
        )
    )
    prediction_assets_saved = False
    for model_name in model_names:
        model = classical_model(model_name, config.seed)
        before = process.memory_info().rss
        fit_started = time.perf_counter()
        model.fit(train_values, dataset.target[split["train"]])
        training_time = time.perf_counter() - fit_started
        inference_started = time.perf_counter()
        validation_score = decision_scores(model, validation_values)
        test_score = decision_scores(model, test_values)
        inference_time = time.perf_counter() - inference_started
        peak_mb = max(before, process.memory_info().rss) / (1024**2)
        metrics, calibrator, threshold, probability = _evaluate(
            model_name=model_name,
            validation_target=dataset.target[split["validation"]],
            validation_score=validation_score,
            test_target=dataset.target[split["test"]],
            test_score=test_score,
            training_time=training_time,
            inference_time=inference_time,
            peak_memory_mb=peak_mb,
            threshold_strategy=threshold_strategy,
            bootstrap_iterations=bootstrap_iterations,
            seed=config.seed,
        )
        metrics.update(
            {
                "dataset_track": "uci_diagnostic",
                "prediction_type": "heart-disease presence classification",
                "input_representation": "consensus_selected_standardized_original_fields",
                "feature_count": len(selected),
                "trainable_parameter_count": None,
                "seed": config.seed,
                "circuit_kernel_evaluations": 0,
            }
        )
        metrics_records.append(metrics)
        joblib.dump(model, root / "models" / f"{model_name}.joblib")
        joblib.dump(calibrator, root / "calibration" / f"{model_name}_platt.joblib")
        _save_predictions(
            root / "predictions" / f"{model_name}.csv",
            dataset.patient_ids[split["test"]],
            dataset.target[split["test"]],
            test_score,
            probability,
            threshold,
        )
        _write_json(
            root / "explanations" / f"{model_name}_permutation.json",
            permutation_explanation(
                model,
                test_values,
                dataset.target[split["test"]],
                selected,
                seed=config.seed,
            ),
        )
        if model_name == "logistic" and not prediction_assets_saved:
            joblib.dump(model, root / "models" / "prediction_model.joblib")
            joblib.dump(calibrator, root / "calibration" / "prediction_calibrator.joblib")
            _write_json(
                root / "models" / "prediction_metadata.json",
                {"model": model_name, "run_id": run_id, "threshold": threshold},
            )
            prediction_assets_saved = True

    resources = config.section("resources")
    subset_limit = int(config.section("training").get("quantum_train_limit", 50))
    quantum_indices = _balanced_subset(split["train"], dataset.target, subset_limit, config.seed)
    local_positions = np.searchsorted(split["train"], quantum_indices)
    _write_json(
        root / "split_manifest_quantum_subset.json",
        {
            "patient_ids": dataset.patient_ids[quantum_indices],
            "target": dataset.target[quantum_indices],
            "selection": "balanced training-only subset",
        },
    )
    q_train = train_quantum[local_positions]
    y_q_train = dataset.target[quantum_indices]
    expected_kernel_evaluations = (
        len(q_train) * (len(q_train) + 1) // 2
        + len(validation_quantum) * len(q_train)
        + len(test_quantum) * len(q_train)
    )
    if expected_kernel_evaluations > int(resources.get("max_kernel_evaluations", 20_000)):
        raise RuntimeError(
            f"estimated kernel evaluations {expected_kernel_evaluations} exceed configured limit"
        )
    kernel = PauliStatevectorKernel(
        len(selected),
        reps=int(config.section("qsvm").get("feature_map_reps", 2)),
        entanglement=str(config.section("qsvm").get("entanglement", "linear")),
        seed=config.seed,
    )
    before = process.memory_info().rss
    fit_started = time.perf_counter()
    train_kernel = kernel.matrix(q_train, q_train, symmetric=True)
    validation_kernel = kernel.matrix(validation_quantum, q_train)
    qsvm_section = config.section("qsvm")
    candidates = [
        float(value) for value in qsvm_section.get("C_candidates", [qsvm_section.get("C", 1.0)])
    ]
    tuning_records: list[dict[str, float]] = []
    qsvm = None
    best_objective = -float("inf")
    for candidate in candidates:
        candidate_model = SVC(kernel="precomputed", C=candidate, class_weight="balanced").fit(
            train_kernel, y_q_train
        )
        candidate_score = candidate_model.decision_function(validation_kernel)
        objective = float(
            average_precision_score(dataset.target[split["validation"]], candidate_score)
        )
        tuning_records.append({"C": candidate, "validation_auprc": objective})
        if objective > best_objective:
            best_objective = objective
            qsvm = candidate_model
    if qsvm is None:  # pragma: no cover - configuration validation prevents empty candidates
        raise RuntimeError("QSVM tuning produced no candidate model")
    _write_json(
        root / "metrics" / "qsvm_validation_tuning.json",
        {
            "objective": "validation_auprc",
            "test_set_evaluated": False,
            "candidates": tuning_records,
            "selected_C": float(qsvm.C),
        },
    )
    training_time = time.perf_counter() - fit_started
    inference_started = time.perf_counter()
    test_kernel = kernel.matrix(test_quantum, q_train)
    validation_score = qsvm.decision_function(validation_kernel)
    test_score = qsvm.decision_function(test_kernel)
    inference_time = time.perf_counter() - inference_started
    np.savez_compressed(
        root / "models" / "oqsvm_kernel_cache.npz",
        train=train_kernel,
        validation=validation_kernel,
        test=test_kernel,
    )
    metrics, calibrator, threshold, probability = _evaluate(
        model_name="oqsvm_pauli",
        validation_target=dataset.target[split["validation"]],
        validation_score=validation_score,
        test_target=dataset.target[split["test"]],
        test_score=test_score,
        training_time=training_time,
        inference_time=inference_time,
        peak_memory_mb=max(before, process.memory_info().rss) / (1024**2),
        threshold_strategy=threshold_strategy,
        bootstrap_iterations=bootstrap_iterations,
        seed=config.seed,
    )
    target_outer = np.outer(2 * y_q_train - 1, 2 * y_q_train - 1)
    target_alignment = np.sum(train_kernel * target_outer) / (
        np.linalg.norm(train_kernel) * np.linalg.norm(target_outer)
    )
    qsvm_evaluations = kernel.execution_count
    metrics.update(
        {
            "dataset_track": "uci_diagnostic",
            "prediction_type": "heart-disease presence classification",
            "input_representation": "consensus_selected_quantum_angle_scaled",
            "feature_count": len(selected),
            "trainable_parameter_count": 0,
            "seed": config.seed,
            "circuit_kernel_evaluations": kernel.execution_count,
            "unique_training_kernel_pairs": len(q_train) * (len(q_train) + 1) // 2,
            "support_vectors_by_class": qsvm.n_support_.astype(int).tolist(),
            "kernel_target_alignment": float(target_alignment),
            "quantum_resources": kernel.resource_summary(),
            "implementation_note": "PennyLane Pauli Z/ZZ fidelity kernel; Qiskit ML unavailable",
            "selected_C": float(qsvm.C),
            "selection_objective": "validation_auprc",
        }
    )
    metrics_records.append(metrics)
    joblib.dump(qsvm, root / "models" / "oqsvm.joblib")
    joblib.dump(calibrator, root / "calibration" / "oqsvm_platt.joblib")
    _save_predictions(
        root / "predictions" / "oqsvm_pauli.csv",
        dataset.patient_ids[split["test"]],
        dataset.target[split["test"]],
        test_score,
        probability,
        threshold,
    )
    _write_json(
        root / "explanations" / "oqsvm_feature_ablation.json",
        perturbation_explanation(
            lambda values: qsvm.decision_function(kernel.matrix(values, q_train)),
            test_quantum[: min(8, len(test_quantum))],
            selected,
        ),
    )

    qmlp_configs = list(
        config.section("models").get(
            "qmlp_variants",
            [
                {"name": "hqmlp_preferred", "ansatz": "ry_rz_cnot", "loss": "bce_logits"},
                {"name": "hqmlp_paper_ablation", "ansatz": "u3_cnot", "loss": "mse_paper_ablation"},
            ],
        )
    )
    for variant in qmlp_configs:
        name = str(variant["name"])
        model = HybridQMLP(
            len(selected),
            qubits=len(selected),
            layers=int(config.section("quantum").get("layers", 1)),
            ansatz=str(variant.get("ansatz", "ry_rz_cnot")),
            entanglement=str(
                variant.get("entanglement", config.section("qmlp").get("entanglement", "linear"))
            ),
            dropout=float(config.section("qmlp").get("dropout", 0.0)),
            init_scale=float(config.section("qmlp").get("init_scale", 0.05)),
            seed=config.seed,
        )
        before = process.memory_info().rss
        training = train_qmlp(
            model,
            train_values[local_positions],
            y_q_train,
            validation_values,
            dataset.target[split["validation"]],
            epochs=int(config.section("training").get("qmlp_epochs", 3)),
            learning_rate=float(config.section("qmlp").get("learning_rate", 0.01)),
            weight_decay=float(config.section("qmlp").get("weight_decay", 0.0)),
            loss_name=str(variant.get("loss", "bce_logits")),
            patience=int(config.section("qmlp").get("early_stopping_patience", 2)),
        )
        inference_started = time.perf_counter()
        validation_score = qmlp_scores(model, validation_values)
        test_score = qmlp_scores(model, test_values)
        inference_time = time.perf_counter() - inference_started
        metrics, calibrator, threshold, probability = _evaluate(
            model_name=name,
            validation_target=dataset.target[split["validation"]],
            validation_score=validation_score,
            test_target=dataset.target[split["test"]],
            test_score=test_score,
            training_time=training.training_time_seconds,
            inference_time=inference_time,
            peak_memory_mb=max(before, process.memory_info().rss) / (1024**2),
            threshold_strategy=threshold_strategy,
            bootstrap_iterations=bootstrap_iterations,
            seed=config.seed,
        )
        summary = model.resource_summary()
        metrics.update(
            {
                "dataset_track": "uci_diagnostic",
                "prediction_type": "heart-disease presence classification",
                "input_representation": "consensus_selected_standardized_hybrid_projection",
                "feature_count": len(selected),
                "trainable_parameter_count": summary["total_trainable_parameters"],
                "seed": config.seed,
                "circuit_kernel_evaluations": model.forward_circuit_evaluations,
                "quantum_resources": summary,
                "training_tracking": training.__dict__,
                "loss": str(variant.get("loss", "bce_logits")),
            }
        )
        metrics_records.append(metrics)
        torch.save(model.state_dict(), root / "models" / f"{name}.pt")
        _write_json(
            root / "models" / f"{name}_architecture.json",
            {
                "input_features": len(selected),
                "qubits": len(selected),
                "layers": model.layers,
                "ansatz": model.ansatz,
                "entanglement": model.entanglement,
                "resources": summary,
            },
        )
        joblib.dump(calibrator, root / "calibration" / f"{name}_platt.joblib")
        _save_predictions(
            root / "predictions" / f"{name}.csv",
            dataset.patient_ids[split["test"]],
            dataset.target[split["test"]],
            test_score,
            probability,
            threshold,
        )
        _write_json(
            root / "explanations" / f"{name}_feature_ablation.json",
            perturbation_explanation(
                lambda values, fitted=model: qmlp_scores(fitted, values),
                test_values[: min(8, len(test_values))],
                selected,
            ),
        )

    _write_json(root / "metrics" / "model_metrics.json", metrics_records)
    pd.DataFrame(
        [
            {
                key: value
                for key, value in record.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            }
            for record in metrics_records
        ]
    ).to_csv(root / "metrics" / "model_metrics.csv", index=False)
    _plot_metrics(metrics_records, root / "plots" / "held_out_auroc_auprc.png")
    _write_json(
        root / "scaling" / "kernel_cost.json",
        {
            "formula": "n_train(n_train+1)/2",
            "quantum_train_patients": len(q_train),
            "unique_training_pairs": len(q_train) * (len(q_train) + 1) // 2,
            "estimated_total_evaluations": expected_kernel_evaluations,
            "actual_total_evaluations_before_explanations": qsvm_evaluations,
        },
    )

    example_record = {
        name: float(dataset.frame.iloc[split["test"][0]][name]) for name in UCI_FEATURES
    }
    prediction = predict_record(root, example_record, explain=True)
    _write_json(root / "predictions" / "held_out_prediction_example.json", prediction)
    best = max(metrics_records, key=lambda item: item["auprc"])
    elapsed = time.perf_counter() - started
    summary = {
        "run_id": run_id,
        "status": "completed",
        "track": "uci_diagnostic",
        "prediction_type": "heart-disease presence classification",
        "selected_features": selected,
        "split_counts": dataset_report(dataset, split)["split_counts"],
        "models": [record["model"] for record in metrics_records],
        "best_test_auprc_model_descriptive_only": best["model"],
        "best_test_auprc_descriptive_only": best["auprc"],
        "runtime_seconds": elapsed,
        "framingham_status": "awaiting_official_teaching_dataset",
        "claim_boundary": (
            "Small single-seed statevector smoke run; no clinical validation or computational "
            "quantum advantage is established. Test metrics are final descriptive evidence, not "
            "a tuning objective."
        ),
    }
    _write_json(root / "summary.json", summary)
    (root / "summary.md").write_text(
        "# UCI IHD QML smoke summary\n\n"
        "**RESEARCH/EDUCATIONAL EXPERIMENT - NOT A MEDICAL DIAGNOSIS**\n\n"
        f"- Run: `{run_id}`\n"
        f"- Split: `{summary['split_counts']}`\n"
        f"- Selected fields: `{', '.join(selected)}`\n"
        f"- Models completed: `{', '.join(summary['models'])}`\n"
        f"- Runtime: `{elapsed:.2f}` seconds\n"
        "- Framingham: official teaching data was not locally available; no result was "
        "fabricated.\n"
        "- Interpretation: proof of functionality only. See `metrics/model_metrics.json` "
        "for exact results.\n",
        encoding="utf-8",
    )
    return summary
