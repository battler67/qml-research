"""Reproduce the tutorial order, then show the leakage-corrected equivalent."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qml_research.quantum import QuantumKernel


def evaluate(X_train, X_test, y_train, y_test, seed: int = 42):
    kernel = QuantumKernel(X_train.shape[1], feature_map="angle", seed=seed)
    train = kernel.matrix(X_train, X_train, symmetric=True)
    test = kernel.matrix(X_test, X_train)
    model = SVC(kernel="precomputed").fit(train, y_train)
    return {
        "accuracy": float(accuracy_score(y_test, model.predict(test))),
        "circuit_executions": kernel.execution_count,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "kernel_quality": kernel.quality(train).to_dict(),
    }


def main() -> None:
    X, y = load_iris(return_X_y=True)
    X = X[:100]
    y = y[:100]

    # Published tutorial ordering, retained only as a reproducibility reference.
    globally_scaled = StandardScaler().fit_transform(X)
    split = train_test_split(globally_scaled, y, test_size=0.25, random_state=42, stratify=y)
    tutorial_order = evaluate(*split)

    # Correct comparison: split first, fit the scaler on training data only.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    corrected = evaluate(scaler.transform(X_train), scaler.transform(X_test), y_train, y_test)
    result = {
        "dataset": "binary Iris sanity check",
        "warning": "Perfect accuracy is not medical evidence or quantum advantage.",
        "tutorial_preprocessing_order": tutorial_order,
        "fold_local_preprocessing": corrected,
    }
    output = Path("results/reproductions/pennylane_tutorial_reproduction.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
