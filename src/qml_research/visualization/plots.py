"""Regenerate the required Phase 1 figures from raw JSON evidence."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from qml_research.results import load_records, records_to_frame


def _save(fig: plt.Figure, output: Path, name: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / name
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _empty(title: str, message: str = "Insufficient completed records") -> plt.Figure:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    return fig


def _completed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.loc[frame["status"] == "completed"].copy()


def _primary_wdbc_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if record.get("status") == "completed"
        and record.get("experiment") == "wdbc_phase1_checkpoint"
        and record.get("run", {}).get("sample_size") == 200
        and record.get("run", {}).get("reduction") == "pca"
        and record.get("run", {}).get("feature_count") == 4
    ]
    return selected or [record for record in records if record.get("status") == "completed"]


def _evidence_caption(frame: pd.DataFrame) -> str:
    data = _completed(frame)
    if data.empty:
        return "No completed evidence records"
    datasets = ", ".join(sorted(str(value) for value in data["dataset"].dropna().unique()))
    models = ", ".join(sorted(str(value) for value in data["model"].dropna().unique()))
    seeds = ", ".join(
        str(int(value)) if float(value).is_integer() else str(value)
        for value in sorted(data["seed"].dropna().unique())
    )
    features = ", ".join(sorted({str(value) for value in data["feature_count"].dropna().unique()}))
    backends = sorted(str(value) for value in data["backend"].dropna().unique())
    backend_text = ", ".join(["classical CPU", *backends])
    return (
        f"Evidence scope — datasets: {datasets}; models: {models}; seeds: {seeds}; "
        f"features/qubits: {features}; backends: {backend_text}. "
        "Titles identify representative versus aggregated views."
    )


def _metric_comparison(frame: pd.DataFrame) -> plt.Figure:
    data = _completed(frame).dropna(subset=["balanced_accuracy"])
    if data.empty:
        return _empty("Classical versus quantum balanced accuracy")
    summary = data.groupby(["dataset", "model"], as_index=False)["balanced_accuracy"].mean()
    fig, axis = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=summary, x="model", y="balanced_accuracy", hue="dataset", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Model")
    axis.set_ylabel("Mean balanced accuracy")
    axis.set_title("Classical versus quantum metric comparison (available folds)")
    axis.tick_params(axis="x", rotation=25)
    return fig


def _confusion(records: list[dict[str, Any]]) -> plt.Figure:
    completed = _primary_wdbc_records(records)
    models = sorted({record.get("run", {}).get("model") for record in completed})
    if not models:
        return _empty("Confusion matrices")
    columns = min(3, len(models))
    rows = int(np.ceil(len(models) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.3 * columns, 4 * rows), squeeze=False)
    for axis, model in zip(axes.flat, models, strict=False):
        matrices = [
            np.asarray(record["metrics"]["confusion_matrix"], dtype=int)
            for record in completed
            if record.get("run", {}).get("model") == model
        ]
        matrix = np.sum(matrices, axis=0)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set_title(f"{model} (WDBC, n=200, seed 42, fold 0, PCA-4)")
        axis.set_xlabel("Predicted: 0 no disease, 1 disease")
        axis.set_ylabel("Actual: 0 no disease, 1 disease")
    for axis in axes.flat[len(models) :]:
        axis.axis("off")
    fig.suptitle("WDBC confusion matrices by matched model", y=1.01)
    return fig


def _curve_plot(records: list[dict[str, Any]], curve_name: str) -> plt.Figure:
    completed = _primary_wdbc_records(records)
    fig, axis = plt.subplots(figsize=(7, 6))
    drawn = False
    seen: set[str] = set()
    for record in completed:
        model = record.get("run", {}).get("model", "unknown")
        if model in seen:
            continue
        curve = record.get("metrics", {}).get(curve_name, {})
        if curve_name == "roc_curve":
            x, y = curve.get("fpr", []), curve.get("tpr", [])
            x_label, y_label = "False-positive rate", "True-positive rate"
        else:
            x, y = curve.get("recall", []), curve.get("precision", [])
            x_label, y_label = "Recall/sensitivity", "Precision"
        if len(x) and len(y):
            axis.plot(x, y, label=model)
            seen.add(model)
            drawn = True
    if not drawn:
        plt.close(fig)
        return _empty("ROC curves" if curve_name == "roc_curve" else "Precision-recall curves")
    if curve_name == "roc_curve":
        axis.plot([0, 1], [0, 1], "--", color="grey", label="chance")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.set_title(
        ("ROC" if curve_name == "roc_curve" else "Precision-recall")
        + " curves (WDBC n=200, seed 42, fold 0, PCA-4 per model)"
    )
    axis.legend()
    return fig


def _vqc_loss(records: list[dict[str, Any]]) -> plt.Figure:
    for record in records:
        loss = record.get("training_history", {}).get("loss", [])
        if record.get("status") == "completed" and loss:
            run = record["run"]
            fig, axis = plt.subplots(figsize=(8, 5))
            axis.plot(np.arange(1, len(loss) + 1), loss)
            axis.set_xlabel("Optimization step")
            axis.set_ylabel("Binary cross-entropy")
            axis.set_title(
                f"VQC training loss: {run['dataset']}, seed {run['seed']}, "
                f"{run['feature_count']} features, {run['vqc_layers']} layer(s)"
            )
            return fig
    return _empty("VQC training-loss curve")


def _kernel_heatmap(records: list[dict[str, Any]], result_root: Path) -> plt.Figure:
    for record in records:
        artifact = record.get("artifacts", {}).get("kernel_matrix")
        if record.get("status") != "completed" or not artifact:
            continue
        path = Path(artifact)
        if not path.is_absolute() and not path.exists():
            candidate = result_root / path.name
            path = candidate if candidate.exists() else path
        if path.exists():
            matrix = np.load(path)["train"]
            run = record["run"]
            fig, axis = plt.subplots(figsize=(7, 6))
            sns.heatmap(matrix, cmap="viridis", vmin=0, vmax=1, ax=axis)
            axis.set_xlabel("Training sample")
            axis.set_ylabel("Training sample")
            axis.set_title(
                f"Quantum Gram matrix: {run['dataset']}, {run['feature_map']}, "
                f"seed {run['seed']}, {run['feature_count']} features"
            )
            return fig
    return _empty("Quantum kernel Gram-matrix heatmap")


def _line_by(frame: pd.DataFrame, x: str, title: str, x_label: str) -> plt.Figure:
    data = _completed(frame).dropna(subset=[x, "balanced_accuracy"])
    if data.empty:
        return _empty(title)
    data = data.copy()
    data[x] = pd.to_numeric(data[x], errors="coerce")
    data = data.dropna(subset=[x])
    if data.empty:
        return _empty(title)
    summary = data.groupby([x, "model"], as_index=False)["balanced_accuracy"].mean()
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    sns.lineplot(data=summary, x=x, y="balanced_accuracy", hue="model", marker="o", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Mean balanced accuracy")
    axis.set_title(title + " (available folds/seeds)")
    return fig


def _circuit_evaluations(frame: pd.DataFrame) -> plt.Figure:
    data = _completed(frame).dropna(subset=["circuit_executions", "sample_size"])
    if data.empty:
        return _empty("Circuit evaluations versus dataset size")
    data = data.loc[data["sample_size"] != "full"].copy()
    data["sample_size"] = pd.to_numeric(data["sample_size"], errors="coerce")
    data = data.dropna(subset=["sample_size"])
    if data.empty:
        return _empty("Circuit evaluations versus dataset size")
    summary = data.groupby(["sample_size", "model"], as_index=False)["circuit_executions"].mean()
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    sns.lineplot(
        data=summary,
        x="sample_size",
        y="circuit_executions",
        hue="model",
        marker="o",
        ax=axis,
    )
    axis.set_xlabel("Stratified dataset size")
    axis.set_ylabel("Mean quantum-circuit executions")
    axis.set_title("Circuit evaluations versus dataset size (available folds/seeds)")
    return fig


def _noise(frame: pd.DataFrame) -> plt.Figure:
    data = _completed(frame)
    data = data.loc[data["experiment"] == "wdbc_kernel_noise"].dropna(
        subset=["quantum_condition", "balanced_accuracy"]
    )
    if data.empty or data["quantum_condition"].nunique() < 2:
        return _empty("Ideal versus noisy performance")
    metrics = ["balanced_accuracy", "sensitivity", "specificity", "auroc"]
    melted = data.melt(
        id_vars=["quantum_condition"], value_vars=metrics, var_name="metric", value_name="value"
    )
    fig, axis = plt.subplots(figsize=(9, 5.5))
    sns.barplot(data=melted, x="metric", y="value", hue="quantum_condition", ax=axis)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Metric")
    axis.set_ylabel("Value")
    axis.set_title("WDBC n=100, seed 42, fold 0: ideal versus noisy Angle QKSVM")
    return fig


def _circuits(records: list[dict[str, Any]]) -> plt.Figure:
    selected: list[tuple[str, str]] = []
    for record in records:
        text = record.get("artifacts", {}).get("circuit_text")
        run = record.get("run", {})
        if record.get("status") != "completed" or not text:
            continue
        label = f"{run.get('model')} / {run.get('feature_map') or 'angle'}"
        if label not in {item[0] for item in selected}:
            selected.append((label, text))
        if len(selected) == 3:
            break
    if not selected:
        return _empty("QML circuit diagrams")
    fig, axes = plt.subplots(len(selected), 1, figsize=(13, 3.4 * len(selected)), squeeze=False)
    for axis, (label, diagram) in zip(axes.flat, selected, strict=True):
        axis.axis("off")
        axis.set_title(label)
        axis.text(0.01, 0.98, diagram, va="top", family="monospace", fontsize=7)
    fig.suptitle("Implemented QML architectures", y=1.01)
    return fig


def generate_all_plots(
    results_root: str | Path = "results/raw",
    output_dir: str | Path = "results/figures",
) -> list[Path]:
    root = Path(results_root)
    output = Path(output_dir)
    records = load_records(root)
    frame = records_to_frame(records)
    feature_frame = frame.loc[frame["experiment"] == "wdbc_feature_kernel_checkpoint"]
    size_frame = frame.loc[
        (frame["experiment"] == "wdbc_phase1_checkpoint")
        & (frame["preprocessing"] == "fold_local_quantum_matched")
    ]
    caption = _evidence_caption(frame)
    plots: list[tuple[str, Callable[[], plt.Figure]]] = [
        ("01_metric_comparison.png", lambda: _metric_comparison(frame)),
        ("02_confusion_matrices.png", lambda: _confusion(records)),
        ("03_roc_curves.png", lambda: _curve_plot(records, "roc_curve")),
        ("04_precision_recall_curves.png", lambda: _curve_plot(records, "precision_recall_curve")),
        ("05_vqc_training_loss.png", lambda: _vqc_loss(records)),
        ("06_quantum_kernel_heatmap.png", lambda: _kernel_heatmap(records, root)),
        (
            "07_performance_vs_features.png",
            lambda: _line_by(
                feature_frame,
                "feature_count",
                "WDBC n=100, seed 42, fold 0: performance versus selected features",
                "Selected features/qubits",
            ),
        ),
        (
            "08_performance_vs_dataset_size.png",
            lambda: _line_by(
                size_frame,
                "sample_size",
                "WDBC seed 42, fold 0: performance versus dataset size",
                "Stratified dataset size",
            ),
        ),
        ("09_circuit_evaluations_vs_size.png", lambda: _circuit_evaluations(size_frame)),
        ("10_ideal_vs_noisy.png", lambda: _noise(frame)),
        ("11_qml_circuits.png", lambda: _circuits(records)),
    ]
    figures: list[Path] = []
    for filename, factory in plots:
        figure = factory()
        figure.text(0.5, -0.025, caption, ha="center", va="top", fontsize=6, wrap=True)
        figures.append(_save(figure, output, filename))
    return figures
