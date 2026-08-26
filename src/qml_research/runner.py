"""Configuration-driven, matched-fold experiment orchestration."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.svm import SVC

from qml_research.classical import build_classical_model, prediction_scores
from qml_research.config import ExperimentConfig
from qml_research.data import load_dataset, make_stratified_folds, stratified_subsample
from qml_research.evaluation import classification_metrics
from qml_research.preprocessing import make_full_preprocessor, make_preprocessor
from qml_research.quantum import QuantumKernel, VariationalQuantumClassifier
from qml_research.results import ResultStore, git_state, load_records

CLASSICAL_MODELS = {"logistic_regression", "linear_svm", "rbf_svm", "random_forest"}


def _timed_fit_predict(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
    started = perf_counter()
    model.fit(X_train, y_train)
    training_time = perf_counter() - started
    started = perf_counter()
    y_pred = np.asarray(model.predict(X_test), dtype=int)
    y_score = prediction_scores(model, X_test)
    prediction_time = perf_counter() - started
    return y_pred, y_score, training_time, prediction_time


def _base_record(
    config: ExperimentConfig,
    dataset: Any,
    run: dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    return {
        "experiment": config.name,
        "status": "running",
        "run": run,
        "dataset_provenance": dataset.provenance,
        "positive_class": dataset.positive_class,
        "task_description": dataset.task_description,
        "split": {
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
            "train_size": int(len(train_indices)),
            "test_size": int(len(test_indices)),
        },
        "git": git_state(),
    }


def _write_failure(
    store: ResultStore,
    run_id: str,
    record: dict[str, Any],
    error: BaseException,
) -> None:
    store.write(
        run_id,
        {
            **record,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": "".join(traceback.format_exception(error))[-8000:],
        },
    )


def _write_skip(
    store: ResultStore,
    run_id: str,
    record: dict[str, Any],
    reason: str,
    projected_runtime_seconds: float | None,
) -> None:
    store.write(
        run_id,
        {
            **record,
            "status": "skipped",
            "skip_reason": reason,
            "projected_runtime_seconds": projected_runtime_seconds,
        },
    )


def _project_full_qml_runtime(
    store: ResultStore,
    config: ExperimentConfig,
    run: dict[str, Any],
    full_sample_count: int,
) -> float | None:
    matching_times: list[float] = []
    identity_fields = [
        "dataset",
        "seed",
        "model",
        "reduction",
        "feature_count",
        "feature_map",
        "vqc_layers",
        "quantum_condition",
    ]
    for record in load_records(store.root):
        previous = record.get("run", {})
        if record.get("status") != "completed" or record.get("experiment") != config.name:
            continue
        if previous.get("sample_size") != 200:
            continue
        if any(previous.get(field) != run.get(field) for field in identity_fields):
            continue
        metrics = record.get("metrics", {})
        elapsed = float(metrics.get("training_time_seconds", 0.0)) + float(
            metrics.get("prediction_time_seconds", 0.0)
        )
        if elapsed > 0:
            matching_times.append(elapsed)
    if not matching_times:
        return None
    exponent = 2 if run["model"] == "qksvm" else 1
    scale = (full_sample_count / 200) ** exponent
    return float(np.mean(matching_times) * scale)


def _skip_unbounded_full_qml(
    *,
    config: ExperimentConfig,
    dataset: Any,
    store: ResultStore,
    run: dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> bool:
    if run.get("sample_size") != "full" or run.get("model") not in {"qksvm", "vqc"}:
        return False
    projection = _project_full_qml_runtime(store, config, run, len(dataset.y))
    limit = config.max_runtime_hours * 3600
    if projection is not None and projection <= limit:
        return False
    reason = (
        "Full-data QML requires a matching completed 200-sample timing projection."
        if projection is None
        else (
            f"Projected full-data runtime {projection:.1f}s exceeds "
            f"the {limit:.1f}s per-family gate."
        )
    )
    run_id = config.digest(run)
    _write_skip(
        store,
        run_id,
        _base_record(config, dataset, run, train_indices, test_indices),
        reason,
        projection,
    )
    return True


def _execute_once(
    *,
    config: ExperimentConfig,
    dataset: Any,
    store: ResultStore,
    run: dict[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    executor: Callable[[], dict[str, Any]],
    force: bool,
) -> str:
    run_id = config.digest(run)
    if store.exists(run_id) and not force:
        return "reused"
    record = _base_record(config, dataset, run, train_indices, test_indices)
    try:
        evidence = executor()
        store.write(run_id, {**record, "status": "completed", **evidence})
    except Exception as exc:  # every failed scientific run must remain visible
        _write_failure(store, run_id, record, exc)
        return "failed"
    return "completed"


def _classical_evidence(
    model_name: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    model = build_classical_model(model_name, seed)
    y_pred, y_score, training_time, prediction_time = _timed_fit_predict(
        model, X_train, y_train, X_test
    )
    return {
        "metrics": classification_metrics(
            y_test,
            y_pred,
            y_score,
            training_time_seconds=training_time,
            prediction_time_seconds=prediction_time,
        ),
        "predictions": {
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_score": y_score.tolist(),
        },
        "fit": {"fixed_hyperparameters": True},
        "quantum_resources": {},
    }


def _condition_settings(config: ExperimentConfig, condition: str) -> tuple[int | None, float]:
    if condition == "ideal":
        return config.shots, 0.0
    if condition == "finite_shot":
        return config.finite_shots, 0.0
    if condition == "depolarizing":
        return config.finite_shots, config.noise_strength
    raise ValueError(f"unsupported quantum condition: {condition}")


def _qksvm_evidence(
    *,
    store: ResultStore,
    run_id: str,
    feature_map: str,
    condition: str,
    config: ExperimentConfig,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    shots, noise_strength = _condition_settings(config, condition)
    kernel = QuantumKernel(
        n_qubits=X_train.shape[1],
        feature_map=feature_map,
        shots=shots,
        noise_strength=noise_strength,
        backend=config.backend,
        seed=seed,
    )
    started = perf_counter()
    train_kernel = kernel.matrix(X_train, X_train, symmetric=True)
    quality = kernel.quality(train_kernel)
    classifier = SVC(C=1.0, kernel="precomputed", class_weight="balanced", random_state=seed)
    classifier.fit(train_kernel, y_train)
    training_time = perf_counter() - started

    started = perf_counter()
    test_kernel = kernel.matrix(X_test, X_train)
    y_pred = classifier.predict(test_kernel).astype(int)
    y_score = classifier.decision_function(test_kernel).astype(float)
    prediction_time = perf_counter() - started
    matrix_path = store.write_kernel_matrix(run_id, train_kernel, test_kernel)
    resources = kernel.resources(X_train[0])
    resources.update(
        {
            "backend": kernel.device_name,
            "shots": shots,
            "noise_strength": noise_strength,
            "circuit_executions": kernel.execution_count,
            "train_kernel_executions": len(X_train) * (len(X_train) + 1) // 2,
            "test_kernel_executions": len(X_test) * len(X_train),
            "trainable_parameters": 0,
        }
    )
    warnings: list[str] = []
    if quality.negative_eigenvalues:
        warnings.append(
            "Raw Gram matrix has numerical negative eigenvalues; no silent PSD repair applied."
        )
    if condition != "ideal":
        warnings.append("Noisy/finite-shot SVM was retrained because its training kernel changed.")
    return {
        "metrics": classification_metrics(
            y_test,
            y_pred,
            y_score,
            training_time_seconds=training_time,
            prediction_time_seconds=prediction_time,
        ),
        "predictions": {
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_score": y_score.tolist(),
        },
        "kernel_quality": quality.to_dict(),
        "quantum_resources": resources,
        "fit": {"fixed_hyperparameters": True},
        "artifacts": {
            "kernel_matrix": matrix_path.as_posix(),
            "circuit_text": kernel.draw(X_train[0]),
        },
        "warnings": warnings,
    }


def _vqc_evidence(
    *,
    config: ExperimentConfig,
    seed: int,
    layers: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    classifier = VariationalQuantumClassifier(
        n_qubits=X_train.shape[1],
        layers=layers,
        steps=config.vqc_steps,
        batch_size=config.vqc_batch_size,
        learning_rate=config.learning_rate,
        backend=config.backend,
        seed=seed,
    )
    classifier.fit(X_train, y_train)
    training_time = classifier.fit_summary_.training_time_seconds
    started = perf_counter()
    y_score = classifier.decision_function(X_test)
    y_pred = (y_score >= 0.0).astype(int)
    prediction_time = perf_counter() - started
    resources = classifier.resources(X_train[0])
    resources.update(
        {
            "backend": classifier.device_name,
            "shots": None,
            "noise_strength": 0.0,
            "circuit_executions": classifier.execution_count,
        }
    )
    return {
        "metrics": classification_metrics(
            y_test,
            y_pred,
            y_score,
            training_time_seconds=training_time,
            prediction_time_seconds=prediction_time,
        ),
        "predictions": {
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "y_score": y_score.tolist(),
        },
        "quantum_resources": resources,
        "fit": {
            **classifier.fit_summary_.to_dict(),
            "optimizer": config.optimizer,
            "learning_rate": config.learning_rate,
            "steps": config.vqc_steps,
            "batch_size": config.vqc_batch_size,
            "layers": layers,
        },
        "training_history": {"loss": classifier.loss_history_},
        "artifacts": {"circuit_text": classifier.draw(X_train[0])},
        "warnings": []
        if classifier.fit_summary_.converged
        else ["VQC did not meet convergence check."],
    }


def run_experiments(config: ExperimentConfig, *, force: bool = False) -> dict[str, Any]:
    dataset = load_dataset(config.dataset)
    output_root = Path(config.output_dir)
    store = ResultStore(output_root)
    started = perf_counter()
    deadline_seconds = config.max_runtime_hours * 3600
    counts = {"completed": 0, "reused": 0, "failed": 0, "skipped": 0}
    runtime_limit_reached = False

    def account(status: str) -> None:
        counts[status] += 1

    for sample_size in config.sample_sizes:
        for seed in config.seeds:
            subset = stratified_subsample(dataset.y, sample_size, seed)
            X_subset = dataset.X.iloc[subset].reset_index(drop=True)
            y_subset = dataset.y[subset]
            folds = make_stratified_folds(y_subset, config.folds, seed)
            if config.fold_limit is not None:
                folds = folds[: config.fold_limit]

            for fold_number, (train_local, test_local) in enumerate(folds):
                if perf_counter() - started > deadline_seconds:
                    runtime_limit_reached = True
                    break
                X_train_frame = X_subset.iloc[train_local]
                X_test_frame = X_subset.iloc[test_local]
                y_train = y_subset[train_local]
                y_test = y_subset[test_local]
                train_indices = subset[train_local]
                test_indices = subset[test_local]

                classical_models = sorted(set(config.models) & CLASSICAL_MODELS)
                if config.include_full_classical:
                    full_preprocessor = make_full_preprocessor(dataset)
                    X_train_full = np.asarray(
                        full_preprocessor.fit_transform(X_train_frame, y_train), dtype=float
                    )
                    X_test_full = np.asarray(full_preprocessor.transform(X_test_frame), dtype=float)
                    for model_name in classical_models:
                        run = {
                            "dataset": dataset.name,
                            "sample_size": sample_size,
                            "seed": seed,
                            "fold": fold_number,
                            "model": model_name,
                            "reduction": "none",
                            "feature_count": "full",
                            "feature_map": None,
                            "vqc_layers": None,
                            "quantum_condition": None,
                            "preprocessing": "fold_local_full_feature",
                        }
                        status = _execute_once(
                            config=config,
                            dataset=dataset,
                            store=store,
                            run=run,
                            train_indices=train_indices,
                            test_indices=test_indices,
                            executor=partial(
                                _classical_evidence,
                                model_name,
                                seed,
                                X_train_full,
                                y_train,
                                X_test_full,
                                y_test,
                            ),
                            force=force,
                        )
                        account(status)

                for reduction in config.reduction_methods:
                    for feature_count in config.feature_counts:
                        preprocessor = make_preprocessor(dataset, reduction, feature_count, seed)
                        try:
                            X_train = np.asarray(
                                preprocessor.fit_transform(X_train_frame, y_train), dtype=float
                            )
                            X_test = np.asarray(preprocessor.transform(X_test_frame), dtype=float)
                        except Exception as exc:
                            run = {
                                "dataset": dataset.name,
                                "sample_size": sample_size,
                                "seed": seed,
                                "fold": fold_number,
                                "model": "preprocessing",
                                "reduction": reduction,
                                "feature_count": feature_count,
                            }
                            run_id = config.digest(run)
                            _write_failure(
                                store,
                                run_id,
                                _base_record(config, dataset, run, train_indices, test_indices),
                                exc,
                            )
                            account("failed")
                            continue

                        for model_name in classical_models:
                            run = {
                                "dataset": dataset.name,
                                "sample_size": sample_size,
                                "seed": seed,
                                "fold": fold_number,
                                "model": model_name,
                                "reduction": reduction,
                                "feature_count": feature_count,
                                "feature_map": None,
                                "vqc_layers": None,
                                "quantum_condition": None,
                                "preprocessing": "fold_local_quantum_matched",
                            }
                            status = _execute_once(
                                config=config,
                                dataset=dataset,
                                store=store,
                                run=run,
                                train_indices=train_indices,
                                test_indices=test_indices,
                                executor=partial(
                                    _classical_evidence,
                                    model_name,
                                    seed,
                                    X_train,
                                    y_train,
                                    X_test,
                                    y_test,
                                ),
                                force=force,
                            )
                            account(status)

                        if "qksvm" in config.models:
                            for feature_map in config.feature_maps:
                                for condition in config.quantum_conditions:
                                    run = {
                                        "dataset": dataset.name,
                                        "sample_size": sample_size,
                                        "seed": seed,
                                        "fold": fold_number,
                                        "model": "qksvm",
                                        "reduction": reduction,
                                        "feature_count": feature_count,
                                        "feature_map": feature_map,
                                        "vqc_layers": None,
                                        "quantum_condition": condition,
                                        "preprocessing": "fold_local_quantum_matched",
                                    }
                                    run_id = config.digest(run)
                                    if _skip_unbounded_full_qml(
                                        config=config,
                                        dataset=dataset,
                                        store=store,
                                        run=run,
                                        train_indices=train_indices,
                                        test_indices=test_indices,
                                    ):
                                        account("skipped")
                                        continue
                                    status = _execute_once(
                                        config=config,
                                        dataset=dataset,
                                        store=store,
                                        run=run,
                                        train_indices=train_indices,
                                        test_indices=test_indices,
                                        executor=partial(
                                            _qksvm_evidence,
                                            store=store,
                                            run_id=run_id,
                                            feature_map=feature_map,
                                            condition=condition,
                                            config=config,
                                            seed=seed,
                                            X_train=X_train,
                                            y_train=y_train,
                                            X_test=X_test,
                                            y_test=y_test,
                                        ),
                                        force=force,
                                    )
                                    account(status)

                        if "vqc" in config.models:
                            for layers in config.vqc_layers:
                                run = {
                                    "dataset": dataset.name,
                                    "sample_size": sample_size,
                                    "seed": seed,
                                    "fold": fold_number,
                                    "model": "vqc",
                                    "reduction": reduction,
                                    "feature_count": feature_count,
                                    "feature_map": "angle",
                                    "vqc_layers": layers,
                                    "quantum_condition": "ideal",
                                    "preprocessing": "fold_local_quantum_matched",
                                }
                                if _skip_unbounded_full_qml(
                                    config=config,
                                    dataset=dataset,
                                    store=store,
                                    run=run,
                                    train_indices=train_indices,
                                    test_indices=test_indices,
                                ):
                                    account("skipped")
                                    continue
                                status = _execute_once(
                                    config=config,
                                    dataset=dataset,
                                    store=store,
                                    run=run,
                                    train_indices=train_indices,
                                    test_indices=test_indices,
                                    executor=partial(
                                        _vqc_evidence,
                                        config=config,
                                        seed=seed,
                                        layers=layers,
                                        X_train=X_train,
                                        y_train=y_train,
                                        X_test=X_test,
                                        y_test=y_test,
                                    ),
                                    force=force,
                                )
                                account(status)
            if runtime_limit_reached:
                break
        if runtime_limit_reached:
            break

    return {
        "experiment": config.name,
        **counts,
        "runtime_seconds": perf_counter() - started,
        "runtime_limit_reached": runtime_limit_reached,
        "output_dir": str(output_root),
    }
