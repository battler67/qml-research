"""Publication-ready run and summary plots for the QCNN benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_dataset(
    train_images: np.ndarray | None,
    train_labels: np.ndarray,
    output: str | Path,
) -> list[Path]:
    root = Path(output)
    paths: list[Path] = []
    values, counts = np.unique(train_labels, return_counts=True)
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.bar(["normal/benign", "malignant"], [counts[values.tolist().index(i)] for i in (0, 1)])
    axis.set_ylabel("Training samples")
    axis.set_title("Training class distribution")
    paths.append(_save(fig, root / "dataset_class_distribution.png"))
    if train_images is not None:
        fig, axes = plt.subplots(2, 4, figsize=(8, 4))
        for row, label in enumerate((0, 1)):
            candidates = np.flatnonzero(train_labels == label)[:4]
            for column, index in enumerate(candidates):
                axes[row, column].imshow(train_images[index], cmap="gray")
                axes[row, column].axis("off")
                axes[row, column].set_title("malignant" if label else "normal/benign")
        paths.append(_save(fig, root / "dataset_examples.png"))
    return paths


def plot_history(history: list[dict[str, Any]], output: str | Path) -> Path | None:
    if not history:
        return None
    root = Path(output)
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="BCE loss", title="Training curves")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_auroc"] for row in history], label="validation AUROC")
    if "gradient_norm" in history[0]:
        axes[1].plot(epochs, [row["gradient_norm"] for row in history], label="gradient norm")
    axes[1].set(xlabel="Epoch", title="Validation and trainability")
    axes[1].legend()
    return _save(fig, root / "training_history.png")


def plot_evaluation(
    y_true: np.ndarray,
    scores: np.ndarray,
    metrics: dict[str, Any],
    output: str | Path,
) -> list[Path]:
    root = Path(output)
    paths: list[Path] = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    roc = metrics["roc_curve"]
    axes[0].plot(roc["fpr"], roc["tpr"], label=f"AUROC={metrics['auroc']:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False-positive rate", ylabel="Sensitivity", title="ROC curve")
    axes[0].legend()
    precision_recall = metrics["precision_recall_curve"]
    axes[1].plot(
        precision_recall["recall"],
        precision_recall["precision"],
        label=f"AUPRC={metrics['auprc']:.3f}",
    )
    axes[1].set(xlabel="Sensitivity", ylabel="Precision", title="Precision-recall curve")
    axes[1].legend()
    paths.append(_save(fig, root / "roc_precision_recall.png"))

    matrix = np.asarray(metrics["confusion_matrix"])
    fig, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks([0, 1], ["normal/benign", "malignant"])
    axis.set_yticks([0, 1], ["normal/benign", "malignant"])
    axis.set(xlabel="Predicted", ylabel="True", title="Confusion matrix")
    fig.colorbar(image, ax=axis)
    paths.append(_save(fig, root / "confusion_matrix.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(scores[y_true == 0], bins=15, alpha=0.7, label="normal/benign")
    axes[0].hist(scores[y_true == 1], bins=15, alpha=0.7, label="malignant")
    axes[0].axvline(metrics["threshold"], color="black", linestyle="--", label="threshold")
    axes[0].set(xlabel="Malignant probability", ylabel="Count", title="Confidence distribution")
    axes[0].legend()
    probability, observed = calibration_curve(y_true, scores, n_bins=min(8, len(y_true) // 4))
    axes[1].plot(probability, observed, marker="o")
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(
        xlabel="Predicted probability",
        ylabel="Observed malignant fraction",
        title="Calibration",
    )
    paths.append(_save(fig, root / "confidence_calibration.png"))
    return paths


def plot_pca(explained_variance: list[float] | None, output: str | Path) -> Path | None:
    if not explained_variance:
        return None
    values = np.asarray(explained_variance)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.bar(np.arange(1, len(values) + 1), values, label="component")
    axis.plot(np.arange(1, len(values) + 1), np.cumsum(values), marker="o", label="cumulative")
    axis.set(xlabel="PCA component", ylabel="Explained variance ratio", title="PCA variance")
    axis.legend()
    return _save(fig, Path(output) / "pca_explained_variance.png")


def plot_architecture(qubits: int, output: str | Path) -> Path:
    stages = int(np.log2(qubits))
    fig, axis = plt.subplots(figsize=(10, 2.8))
    axis.axis("off")
    labels = [f"{qubits} RY\nencoded qubits"]
    active = qubits
    for _ in range(stages):
        active //= 2
        labels.append(f"shared SU(4)\nconv + pool\n{active} active")
    labels.append("<Z> ->\naffine logit")
    positions = np.linspace(0.08, 0.92, len(labels))
    for index, (position, label) in enumerate(zip(positions, labels, strict=True)):
        axis.text(
            position,
            0.5,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "facecolor": "#e8f0fe", "edgecolor": "#335"},
        )
        if index < len(labels) - 1:
            axis.annotate(
                "",
                xy=(positions[index + 1] - 0.07, 0.5),
                xytext=(position + 0.07, 0.5),
                arrowprops={"arrowstyle": "->"},
            )
    axis.set_title("Hierarchical QCNN architecture")
    return _save(fig, Path(output) / "qcnn_architecture.png")


def plot_summary(records: list[dict[str, Any]], output: str | Path) -> list[Path]:
    completed = [record for record in records if record.get("status") == "completed"]
    if not completed:
        return []
    root = Path(output)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in completed:
        grouped.setdefault((record["dataset"], record["model"]), []).append(record)
    groups = sorted(grouped)
    labels = [f"{dataset}:{model}" for dataset, model in groups]
    auroc = [np.mean([item["metrics"]["auroc"] for item in grouped[key]]) for key in groups]
    balanced = [
        np.mean([item["metrics"]["balanced_accuracy"] for item in grouped[key]]) for key in groups
    ]
    runtime = [
        np.mean([item["metrics"]["training_time_seconds"] for item in grouped[key]])
        for key in groups
    ]
    x = np.arange(len(groups))
    fig, axis = plt.subplots(figsize=(max(8, len(groups) * 1.5), 5))
    axis.bar(x - 0.2, auroc, width=0.4, label="AUROC")
    axis.bar(x + 0.2, balanced, width=0.4, label="Balanced accuracy")
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_ylim(0, 1)
    axis.set_title("Reproduced benchmark metrics")
    axis.legend()
    paths = [_save(fig, root / "model_metric_comparison.png")]
    fig, axis = plt.subplots(figsize=(max(8, len(groups) * 1.5), 4))
    axis.bar(labels, runtime)
    axis.set(ylabel="Training seconds", title="Training runtime comparison")
    axis.tick_params(axis="x", rotation=30)
    paths.append(_save(fig, root / "runtime_comparison.png"))
    return paths


def load_records(root: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((Path(root) / "runs").glob("*/record.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records
