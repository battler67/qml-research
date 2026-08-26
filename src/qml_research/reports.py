"""Regenerate normalized tables and evidence-gated Markdown reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from qml_research.results import load_records, records_to_frame

METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "mcc",
    "auroc",
    "auprc",
]
GROUP_COLUMNS = [
    "experiment",
    "dataset",
    "sample_size",
    "model",
    "reduction",
    "feature_count",
    "feature_map",
    "vqc_layers",
    "quantum_condition",
]
PAIR_COLUMNS = [
    "experiment",
    "dataset",
    "sample_size",
    "seed",
    "fold",
    "reduction",
    "feature_count",
    "preprocessing",
]


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    complete = frame.loc[frame["status"] == "completed"].copy()
    if complete.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in complete.groupby(GROUP_COLUMNS, dropna=False):
        row = dict(zip(GROUP_COLUMNS, keys, strict=True))
        row["runs"] = len(group)
        row["seeds"] = group["seed"].nunique()
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            row[f"{metric}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            if len(values) > 1:
                half = float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))
                row[f"{metric}_ci95_low"] = float(np.mean(values) - half)
                row[f"{metric}_ci95_high"] = float(np.mean(values) + half)
            else:
                row[f"{metric}_ci95_low"] = np.nan
                row[f"{metric}_ci95_high"] = np.nan
        for resource in [
            "training_time_seconds",
            "prediction_time_seconds",
            "circuit_executions",
            "num_qubits",
            "circuit_depth",
            "total_gate_count",
            "two_qubit_gate_count",
            "trainable_parameters",
            "kernel_symmetric_error",
            "kernel_diagonal_max_error",
            "kernel_minimum_eigenvalue",
            "kernel_negative_eigenvalues",
        ]:
            values = pd.to_numeric(group[resource], errors="coerce")
            row[f"{resource}_mean"] = float(values.mean()) if values.notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    complete = frame.loc[frame["status"] == "completed"].copy()
    quantum = complete.loc[complete["model"].isin({"qksvm", "vqc"})]
    classical = complete.loc[
        ~complete["model"].isin({"qksvm", "vqc"})
        & (complete["preprocessing"] == "fold_local_quantum_matched")
    ]
    rows: list[dict[str, Any]] = []
    for _, qml_row in quantum.iterrows():
        candidates = classical
        for column in PAIR_COLUMNS:
            target = qml_row[column]
            mask = candidates[column].isna() if pd.isna(target) else candidates[column] == target
            candidates = candidates.loc[mask]
        for _, classical_row in candidates.iterrows():
            row = {column: qml_row[column] for column in PAIR_COLUMNS}
            row.update(
                {
                    "qml_model": qml_row["model"],
                    "qml_feature_map": qml_row["feature_map"],
                    "qml_vqc_layers": qml_row["vqc_layers"],
                    "quantum_condition": qml_row["quantum_condition"],
                    "classical_model": classical_row["model"],
                }
            )
            for metric in METRIC_COLUMNS:
                qml_value = qml_row[metric]
                classical_value = classical_row[metric]
                row[f"qml_{metric}"] = qml_value
                row[f"classical_{metric}"] = classical_value
                row[f"delta_{metric}"] = (
                    float(qml_value - classical_value)
                    if pd.notna(qml_value) and pd.notna(classical_value)
                    else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: list[str], limit: int = 30) -> str:
    if frame.empty:
        return "_No completed records are available._"
    selected = frame.loc[:, [column for column in columns if column in frame]].head(limit).copy()
    for column in selected.select_dtypes(include=["float"]).columns:
        selected[column] = selected[column].map(
            lambda value: f"{value:.4f}" if pd.notna(value) else "-"
        )
    selected = selected.where(pd.notna(selected), "-")
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in selected.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _best(summary: pd.DataFrame, quantum: bool, dataset: str = "wdbc") -> pd.Series | None:
    if summary.empty:
        return None
    qml_models = {"qksvm", "vqc"}
    mask = summary["model"].isin(qml_models)
    candidates = summary.loc[(mask if quantum else ~mask) & (summary["dataset"] == dataset)]
    if quantum:
        candidates = candidates.loc[
            candidates["quantum_condition"].isna() | (candidates["quantum_condition"] == "ideal")
        ]
    candidates = candidates.dropna(subset=["balanced_accuracy_mean"])
    if candidates.empty:
        return None
    return candidates.sort_values(
        ["balanced_accuracy_mean", "sensitivity_mean", "circuit_executions_mean"],
        ascending=[False, False, True],
        na_position="last",
    ).iloc[0]


def _best_matched_classical(summary: pd.DataFrame, qml_selection: pd.Series) -> pd.Series | None:
    candidates = summary.loc[
        (~summary["model"].isin({"qksvm", "vqc"}))
        & (summary["experiment"] == qml_selection["experiment"])
        & (summary["dataset"] == qml_selection["dataset"])
        & (summary["sample_size"] == qml_selection["sample_size"])
        & (summary["reduction"] == qml_selection["reduction"])
        & (summary["feature_count"] == qml_selection["feature_count"])
    ].dropna(subset=["balanced_accuracy_mean"])
    if candidates.empty:
        return None
    return candidates.sort_values(
        ["balanced_accuracy_mean", "sensitivity_mean"],
        ascending=[False, False],
        na_position="last",
    ).iloc[0]


def _record_matches_summary(record: dict[str, Any], selected: pd.Series) -> bool:
    run = record.get("run", {})
    for column in GROUP_COLUMNS:
        value = record.get("experiment") if column == "experiment" else run.get(column)
        target = selected[column]
        if pd.isna(target):
            if value is not None:
                return False
        elif value != target:
            return False
    return record.get("status") == "completed" and bool(record.get("predictions"))


def _scalar_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray
) -> dict[str, float]:
    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if precision + sensitivity else 0.0
    mcc_denominator = np.sqrt(
        (true_positive + false_positive)
        * (true_positive + false_negative)
        * (true_negative + false_positive)
        * (true_negative + false_negative)
    )
    mcc = (
        (true_positive * true_negative - false_positive * false_negative) / mcc_denominator
        if mcc_denominator
        else 0.0
    )
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "precision": float(precision),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1),
        "mcc": float(mcc),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
    }


def _bootstrap_selected(
    records: list[dict[str, Any]], summary: pd.DataFrame, repetitions: int = 2_000
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    best_qml = _best(summary, quantum=True, dataset="wdbc")
    selections = [
        ("qml", best_qml),
        (
            "classical_matched",
            _best_matched_classical(summary, best_qml) if best_qml is not None else None,
        ),
    ]
    for family, selected in selections:
        if selected is None:
            continue
        matching = [record for record in records if _record_matches_summary(record, selected)]
        if not matching:
            continue
        y_true = np.concatenate(
            [np.asarray(record["predictions"]["y_true"], dtype=int) for record in matching]
        )
        y_pred = np.concatenate(
            [np.asarray(record["predictions"]["y_pred"], dtype=int) for record in matching]
        )
        y_score = np.concatenate(
            [np.asarray(record["predictions"]["y_score"], dtype=float) for record in matching]
        )
        class_indices = [np.flatnonzero(y_true == label) for label in (0, 1)]
        if any(len(indices) == 0 for indices in class_indices):
            continue
        point = _scalar_metrics(y_true, y_pred, y_score)
        distributions = {metric: [] for metric in METRIC_COLUMNS}
        rng = np.random.default_rng(2026)
        for _ in range(repetitions):
            sampled = np.concatenate(
                [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
            )
            metrics = _scalar_metrics(y_true[sampled], y_pred[sampled], y_score[sampled])
            for metric in METRIC_COLUMNS:
                distributions[metric].append(float(metrics[metric]))
        row: dict[str, Any] = {
            "family": family,
            "experiment": selected["experiment"],
            "dataset": selected["dataset"],
            "sample_size": selected["sample_size"],
            "model": selected["model"],
            "reduction": selected["reduction"],
            "feature_count": selected["feature_count"],
            "feature_map": selected["feature_map"],
            "prediction_count": len(y_true),
            "bootstrap_repetitions": repetitions,
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_pooled"] = float(point[metric])
            row[f"{metric}_bootstrap_low"] = float(np.quantile(distributions[metric], 0.025))
            row[f"{metric}_bootstrap_high"] = float(np.quantile(distributions[metric], 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def _noise_kernel_deltas(records: list[dict[str, Any]]) -> pd.DataFrame:
    noise_records = [
        record
        for record in records
        if record.get("status") == "completed"
        and record.get("experiment") == "wdbc_kernel_noise"
        and record.get("run", {}).get("model") == "qksvm"
    ]
    ideal = next(
        (
            record
            for record in noise_records
            if record.get("run", {}).get("quantum_condition") == "ideal"
        ),
        None,
    )
    if ideal is None:
        return pd.DataFrame()
    ideal_path = Path(ideal.get("artifacts", {}).get("kernel_matrix", ""))
    if not ideal_path.is_file():
        return pd.DataFrame()
    with np.load(ideal_path) as matrices:
        ideal_train = np.asarray(matrices["train"], dtype=float)
        ideal_test = np.asarray(matrices["test"], dtype=float)
    rows: list[dict[str, Any]] = []
    for record in noise_records:
        matrix_path = Path(record.get("artifacts", {}).get("kernel_matrix", ""))
        if not matrix_path.is_file():
            continue
        with np.load(matrix_path) as matrices:
            train = np.asarray(matrices["train"], dtype=float)
            test = np.asarray(matrices["test"], dtype=float)
        metrics = record.get("metrics", {})
        quality = record.get("kernel_quality", {})
        resources = record.get("quantum_resources", {})
        rows.append(
            {
                "condition": record["run"]["quantum_condition"],
                "backend": resources.get("backend"),
                "shots": resources.get("shots"),
                "noise_strength": resources.get("noise_strength"),
                "train_kernel_mean_absolute_delta_from_ideal": float(
                    np.mean(np.abs(train - ideal_train))
                ),
                "train_kernel_max_absolute_delta_from_ideal": float(
                    np.max(np.abs(train - ideal_train))
                ),
                "test_kernel_mean_absolute_delta_from_ideal": float(
                    np.mean(np.abs(test - ideal_test))
                ),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "sensitivity": metrics.get("sensitivity"),
                "specificity": metrics.get("specificity"),
                "auroc": metrics.get("auroc"),
                "diagonal_max_error": quality.get("diagonal_max_error"),
                "minimum_eigenvalue": quality.get("minimum_eigenvalue"),
                "negative_eigenvalues": quality.get("negative_eigenvalues"),
                "circuit_executions": resources.get("circuit_executions"),
            }
        )
    return pd.DataFrame(rows)


def generate_reports(
    results_root: str | Path = "results/raw",
    tables_dir: str | Path = "results/tables",
    research_dir: str | Path = "research",
    bootstrap_repetitions: int = 2_000,
) -> dict[str, Path]:
    records = load_records(results_root)
    frame = records_to_frame(records)
    summary = _summary(frame)
    paired = _paired_deltas(frame)
    bootstrap = _bootstrap_selected(records, summary, repetitions=bootstrap_repetitions)
    noise_deltas = _noise_kernel_deltas(records)
    tables = Path(tables_dir)
    research = Path(research_dir)
    tables.mkdir(parents=True, exist_ok=True)
    research.mkdir(parents=True, exist_ok=True)
    normalized_path = tables / "normalized-results.csv"
    summary_path = tables / "summary-metrics.csv"
    failure_path = tables / "failed-runs.csv"
    paired_path = tables / "paired-model-deltas.csv"
    bootstrap_path = tables / "selected-bootstrap-intervals.csv"
    noise_path = tables / "noise-kernel-deltas.csv"
    frame.to_csv(normalized_path, index=False)
    summary.to_csv(summary_path, index=False)
    frame.loc[frame["status"] != "completed"].to_csv(failure_path, index=False)
    paired.to_csv(paired_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    noise_deltas.to_csv(noise_path, index=False)

    complete = frame.loc[frame["status"] == "completed"] if not frame.empty else frame
    results_table = _markdown_table(
        summary.sort_values(["dataset", "balanced_accuracy_mean"], ascending=[True, False])
        if not summary.empty
        else summary,
        [
            "experiment",
            "dataset",
            "sample_size",
            "model",
            "reduction",
            "feature_count",
            "feature_map",
            "quantum_condition",
            "runs",
            "balanced_accuracy_mean",
            "balanced_accuracy_std",
            "sensitivity_mean",
            "specificity_mean",
            "auroc_mean",
            "circuit_executions_mean",
        ],
        limit=max(len(summary), 1),
    )
    paired_table = _markdown_table(
        paired,
        [
            "experiment",
            "dataset",
            "sample_size",
            "seed",
            "fold",
            "feature_count",
            "qml_model",
            "qml_feature_map",
            "classical_model",
            "delta_balanced_accuracy",
            "delta_sensitivity",
            "delta_specificity",
            "delta_auroc",
        ],
        limit=max(len(paired), 1),
    )
    bootstrap_table = _markdown_table(
        bootstrap,
        [
            "family",
            "experiment",
            "model",
            "sample_size",
            "prediction_count",
            "balanced_accuracy_pooled",
            "balanced_accuracy_bootstrap_low",
            "balanced_accuracy_bootstrap_high",
            "sensitivity_pooled",
            "sensitivity_bootstrap_low",
            "sensitivity_bootstrap_high",
        ],
        limit=10,
    )
    quantum_resources = summary.loc[summary["model"].isin({"qksvm", "vqc"})]
    resource_table = _markdown_table(
        quantum_resources,
        [
            "experiment",
            "dataset",
            "sample_size",
            "model",
            "feature_count",
            "feature_map",
            "quantum_condition",
            "num_qubits_mean",
            "circuit_depth_mean",
            "total_gate_count_mean",
            "two_qubit_gate_count_mean",
            "trainable_parameters_mean",
            "circuit_executions_mean",
            "training_time_seconds_mean",
        ],
        limit=max(len(quantum_resources), 1),
    )
    noise_table = _markdown_table(
        noise_deltas,
        [
            "condition",
            "backend",
            "shots",
            "noise_strength",
            "train_kernel_mean_absolute_delta_from_ideal",
            "train_kernel_max_absolute_delta_from_ideal",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "auroc",
            "diagonal_max_error",
            "minimum_eigenvalue",
            "negative_eigenvalues",
        ],
        limit=10,
    )
    failed_count = int((frame["status"] != "completed").sum()) if not frame.empty else 0
    datasets = ", ".join(sorted(complete["dataset"].dropna().unique())) or "none"
    seeds = ", ".join(
        str(int(value)) if float(value).is_integer() else str(value)
        for value in sorted(complete["seed"].dropna().unique())
    )
    seeds = seeds or "none"
    results_text = f"""# Phase 1 Experiment Results

This report is generated from `results/raw/*.json`; values are not hardcoded.

## Coverage

- Completed fold/model records: {len(complete)}
- Failed or skipped records: {failed_count}
- Datasets represented: {datasets}
- Seeds represented: {seeds}

## Results

{results_table}

## Quantum resource telemetry

Circuit structures are expanded at the PennyLane device level. Execution totals are
actual QNode executions counted by the implemented run, not asymptotic estimates.

{resource_table}

## Controlled noise study

Kernel-value deltas are calculated directly from the saved Gram matrices against the
ideal matrix for the identical fixed fold. Unchanged predictions on one small fold do
not establish noise robustness.

{noise_table}

## Paired model deltas

Positive deltas favor QML. Every row uses the same dataset subset, fold, reduction,
feature count, preprocessing and seed as its classical comparator.

{paired_table}

## Selected-configuration bootstrap intervals

These deterministic {bootstrap_repetitions:,}-resample intervals bootstrap saved predictions within
class for the strongest observed WDBC QML and classical configurations. They are
descriptive uncertainty checks; they do not replace repeated cross-validation or
external clinical validation.

{bootstrap_table}

The uncertainty columns in `results/tables/summary-metrics.csv` use two-sided 95%
t-intervals across the available fold records. Repeated-CV folds are correlated, so
these intervals are descriptive and must not be presented as independent clinical
validation.

## Failed and incomplete experiments

Every failure is retained in `results/tables/failed-runs.csv`. Missing planned runs
must be described as incomplete evidence, not silently treated as negative results.

## Interpretation and threats to validity

- These are small public tabular benchmarks and simulator experiments.
- Feature reduction can remove clinically useful classical information.
- Kernel simulation cost is not QPU runtime, and model parameters are not total cost.
- Model selection on WDBC is exploratory; Heart Disease provides a limited transfer
  check, not external clinical validation.
- The checked-in checkpoint evidence uses one seed and one fixed fold to bound
  simulator runtime. The three-seed, five-fold configs are pre-registered but their
  unexecuted combinations must not be described as completed evidence.
- No result here alone establishes quantum advantage or clinical utility.
"""
    results_report_path = research / "experiment-results.md"
    results_report_path.write_text(results_text, encoding="utf-8")

    best_qml = _best(summary, quantum=True, dataset="wdbc")
    best_classical = _best_matched_classical(summary, best_qml) if best_qml is not None else None
    if best_qml is None:
        qml_sentence = "No completed QML configuration is available for selection."
        qml_config = "Pending"
    else:
        model_label = {"qksvm": "QKSVM", "vqc": "VQC"}.get(
            str(best_qml["model"]), str(best_qml["model"])
        )
        feature_map_label = str(best_qml["feature_map"] or "angle").capitalize()
        reduction_label = str(best_qml["reduction"]).upper()
        qml_config = (
            f"{model_label} with {feature_map_label} encoding and "
            f"{best_qml['feature_count']} {reduction_label} features"
        )
        qml_sentence = (
            f"The strongest observed checkpoint QML balanced accuracy is "
            f"{best_qml['balanced_accuracy_mean']:.3f} for {qml_config}."
        )
    if best_classical is None:
        comparison_sentence = "No completed classical comparator is available."
    else:
        comparison_sentence = (
            f"The strongest matched classical balanced accuracy is "
            f"{best_classical['balanced_accuracy_mean']:.3f} for {best_classical['model']} "
            f"on the same subset, fold and reduced features."
        )
    recommendation_text = f"""# QML MVP Recommendation

## Evidence status

{qml_sentence} {comparison_sentence}

**Quantum advantage has not been demonstrated.** Phase 1 uses classically simulated,
small-data circuits and does not establish computational speedup, clinical benefit,
or superiority beyond the tested folds.

## Current integration recommendation

- Primary disease dataset: WDBC for a transparent diagnostic demonstration; retain
  Heart Disease as a cross-dataset robustness check.
- Candidate QML path: {qml_config}.
- Classical references: logistic regression, linear SVM, RBF SVM, and random forest,
  including full-feature results alongside quantum-matched features.
- Simulator: analytic PennyLane `default.qubit`; demonstrate finite-shot and simple
  depolarizing sensitivity separately and label it clearly.
- Phase 2 hardware experiment: freeze preprocessing, the trained model/configuration,
  and a very small inference batch; submit only after an explicit credential and cost
  confirmation gate. Do not perform real-hardware training.

## Quantum Helix Lab integration boundary

Integrate a QML experiment/result view backed by the stable YAML/JSON schema, dataset
provenance, matched classical comparison, resource telemetry, and scientific warnings.
Keep deterministic mutation/search analysis distinct from learned disease prediction.

Postpone clinical uploads, MedMNIST, gene-expression data, automated medical advice,
real-QPU training, and any "early detection" or "quantum advantage" claim.

## Claims we can safely make

- Phase 1 implements reproducible, leakage-aware quantum-kernel and VQC experiments
  with four classical baselines on public Iris, WDBC, and Heart Disease data.
- Selected exploratory configuration: {qml_config}.
- On its fixed WDBC fold, that configuration reached the same balanced accuracy as
  its strongest matched classical comparator.
- Finite shots and depolarizing noise changed the saved Gram matrices even where the
  small fold's predicted classes happened to remain unchanged.
- Circuit executions and device-level resource counts are measured and exposed for
  audit; they are not presented as hardware speedups.

## Claims we must avoid

- Quantum advantage, computational speedup, clinical superiority, or deployment
  readiness.
- Pre-symptomatic or longitudinal "early detection" from WDBC.
- Noise robustness from one fixed fold, or real-hardware performance from simulation.
- Generalization beyond the public benchmark populations or medical advice for an
  individual patient.

## Exact Phase 2 prerequisites

1. Complete all pre-registered seeds/folds or document the resource-limited subset.
2. Review the selected configuration against paired balanced accuracy, sensitivity,
   uncertainty, circuit counts, and noise degradation, not accuracy alone.
3. Freeze a versioned inference contract and reproduce it from a clean environment.
4. Add the adapter to Quantum Helix Lab on a separate integration branch while
   preserving authentication, provider confirmation, and scientific-boundary UI.
"""
    recommendation_path = research / "qml-mvp-recommendation.md"
    recommendation_path.write_text(recommendation_text, encoding="utf-8")
    return {
        "normalized": normalized_path,
        "summary": summary_path,
        "failures": failure_path,
        "paired_deltas": paired_path,
        "bootstrap_intervals": bootstrap_path,
        "noise_kernel_deltas": noise_path,
        "results_report": results_report_path,
        "recommendation": recommendation_path,
    }
