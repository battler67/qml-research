"""Validated profile configuration for the QCNN experiment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "experiments" / "qcnn_breast_cancer" / "configs"
PROFILE_NAMES = (
    "smoke",
    "laptop_8gb",
    "full_dataset",
    "high_memory",
    "noisy_simulation",
    "real_hardware_ready",
)

SUPPORTED_MODELS = {
    "qcnn",
    "logistic",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "reduced_mlp",
    "small_cnn",
    "resnet18_random",
    "resnet18_imagenet",
    "mobilenet_v3_small_imagenet",
}


@dataclass(frozen=True)
class QCNNConfig:
    experiment_name: str = "qcnn_breast_cancer"
    dataset: str = "breastmnist"
    dataset_version: str = "3.0.2"
    data_dir: str = "data/raw/medmnist"
    output_dir: str = "results/qcnn_breast_cancer"
    train_size: int | str = "full"
    validation_size: int | str = "full"
    test_size: int | str = "full"
    samples_per_class: int | None = None
    stratified: bool = True
    image_size: int = 28
    reducer: str = "spatial_pool"
    reduced_features: int = 4
    qubits: int = 4
    encoding: str = "angle_ry"
    stages: int = 2
    convolution: str = "su4_ising"
    pooling: str = "unitary_controlled"
    models: list[str] = field(default_factory=lambda: ["qcnn", "logistic"])
    backend: str = "lightning.qubit"
    gradient_method: str = "adjoint"
    shots: int | None = None
    noise_enabled: bool = False
    noise_probabilities: list[float] = field(default_factory=list)
    shot_values: list[int] = field(default_factory=list)
    optimizer: str = "adam"
    learning_rate: float = 0.01
    image_learning_rate: float = 0.001
    fine_tune_learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    batch_size: int = 16
    image_batch_size: int = 32
    epochs: int = 30
    image_epochs: int = 40
    early_stopping_patience: int = 6
    class_weighting: str = "balanced"
    seeds: list[int] = field(default_factory=lambda: [17, 42, 73])
    cv_folds: int = 5
    num_workers: int = 0
    precision: str = "float64"
    pretrained_download: bool = False
    pretrained_mode: str = "frozen"
    threshold_strategy: str = "youden_j"
    bootstrap_iterations: int = 10_000
    cache_enabled: bool = True
    expensive_forward_limit: int = 25_000
    expensive_hour_limit: float = 2.0
    confirm_expensive: bool = False
    real_hardware_provider: str | None = None
    real_hardware_submit: bool = False
    tuning_trials: int = 18
    tuning_space: dict[str, list[Any]] = field(default_factory=dict)
    init_scale: float = 0.05
    profile: str = "laptop_8gb"

    def __post_init__(self) -> None:
        if self.dataset not in {"breastmnist", "wdbc"}:
            raise ValueError("dataset must be breastmnist or wdbc")
        for name in ("train_size", "validation_size", "test_size"):
            value = getattr(self, name)
            if value != "full" and (not isinstance(value, int) or value < 2):
                raise ValueError(f"{name} must be 'full' or an integer >= 2")
        if self.qubits not in {4, 8}:
            raise ValueError("qubits must be 4 or 8")
        expected_stages = int(math.log2(self.qubits))
        if self.stages != expected_stages:
            raise ValueError(f"{self.qubits} qubits require {expected_stages} pooling stages")
        if self.reduced_features != self.qubits:
            raise ValueError("reduced_features must equal qubits for angle encoding")
        if self.reducer not in {"spatial_pool", "pca"}:
            raise ValueError("reducer must be spatial_pool or pca")
        if self.dataset == "wdbc" and self.reducer != "pca":
            raise ValueError("WDBC requires fold-fitted PCA")
        if self.dataset == "wdbc" and self.cv_folds < 2:
            raise ValueError("WDBC requires cv_folds >= 2")
        if not self.stratified:
            raise ValueError("stratified=false is unsupported for this imbalanced benchmark")
        if self.image_size != 28:
            raise ValueError("the verified BreastMNIST pipeline requires image_size=28")
        if self.encoding != "angle_ry":
            raise ValueError("only the preregistered angle_ry encoding is supported")
        if self.convolution != "su4_ising" or self.pooling != "unitary_controlled":
            raise ValueError("unsupported QCNN convolution or pooling implementation")
        if not self.models:
            raise ValueError("models cannot be empty")
        unknown = set(self.models) - SUPPORTED_MODELS
        if unknown:
            raise ValueError(f"unsupported models: {sorted(unknown)}")
        if min(self.batch_size, self.image_batch_size, self.early_stopping_patience) < 1:
            raise ValueError("batch sizes and early_stopping_patience must be positive")
        if self.epochs < 0 or self.image_epochs < 0:
            raise ValueError("epochs cannot be negative")
        if self.optimizer != "adam" or self.class_weighting != "balanced":
            raise ValueError("the verified optimizer/class weighting is adam/balanced")
        if min(self.learning_rate, self.image_learning_rate, self.fine_tune_learning_rate) <= 0:
            raise ValueError("learning rates must be positive")
        if self.weight_decay < 0 or self.init_scale < 0:
            raise ValueError("weight_decay and init_scale cannot be negative")
        if self.num_workers < 0 or self.bootstrap_iterations < 1 or self.tuning_trials < 1:
            raise ValueError("worker, bootstrap, and tuning counts are invalid")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be a non-empty unique list")
        if self.samples_per_class is not None and self.samples_per_class < 1:
            raise ValueError("samples_per_class must be null or positive")
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots must be null or positive")
        if any(value < 0 or value > 1 for value in self.noise_probabilities):
            raise ValueError("noise probabilities must be in [0, 1]")
        if self.real_hardware_submit:
            raise ValueError("real hardware submission is intentionally unsupported")
        if self.pretrained_mode not in {"frozen", "partial_finetune"}:
            raise ValueError("pretrained_mode must be frozen or partial_finetune")
        if self.precision != "float64":
            raise ValueError("the verified quantum precision is float64")
        if self.backend not in {
            "lightning.qubit",
            "default.qubit",
            "default.mixed",
            "hardware_export",
        }:
            raise ValueError("unsupported backend")
        if self.gradient_method not in {"adjoint", "parameter-shift"}:
            raise ValueError("unsupported gradient_method")
        if self.shots is not None and self.gradient_method == "adjoint":
            raise ValueError("finite shots require parameter-shift gradients")
        if self.threshold_strategy not in {"fixed_0.5", "youden_j"}:
            raise ValueError("threshold_strategy must be fixed_0.5 or youden_j")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def estimated_resources(self) -> dict[str, Any]:
        if self.dataset == "breastmnist":
            full_train, full_validation, full_test = 546, 78, 156
            folds = 1
        else:
            largest_outer_test = math.ceil(569 / self.cv_folds)
            largest_train_validation = 569 - math.floor(569 / self.cv_folds)
            full_validation = math.ceil(largest_train_validation * 0.2)
            full_train = largest_train_validation - full_validation
            full_test = largest_outer_test
            folds = self.cv_folds

        def selected(limit: int | str, full_count: int) -> int:
            return full_count if limit == "full" else min(int(limit), full_count)

        train_count = selected(self.train_size, full_train)
        validation_count = selected(self.validation_size, full_validation)
        test_count = selected(self.test_size, full_test)
        forwards_per_fold = (
            self.epochs * (train_count + validation_count) + validation_count + test_count
        )
        forwards = forwards_per_fold * len(self.seeds) * folds if "qcnn" in self.models else 0
        # Conservative calibration from the verified four-qubit Windows CPU smoke run.
        seconds_per_forward = 0.3 if self.qubits == 4 else 1.0
        hours = forwards * seconds_per_forward / 3600
        expensive_reasons: list[str] = []
        if forwards > self.expensive_forward_limit:
            expensive_reasons.append(f"estimated quantum forwards {forwards:,}")
        if hours > self.expensive_hour_limit:
            expensive_reasons.append(f"estimated runtime {hours:.2f} h")
        if self.qubits == 8:
            expensive_reasons.append("eight-qubit profile")
        if self.noise_enabled:
            expensive_reasons.append("noisy/finite-shot sweep")
        return {
            "statevector_complex_amplitudes": 2**self.qubits,
            "statevector_bytes_complex128": (2**self.qubits) * 16,
            "trainable_quantum_parameters": self.stages * 18,
            "trainable_total_parameters": self.stages * 18 + 2,
            "estimated_quantum_forwards_per_fold_seed": forwards_per_fold,
            "estimated_quantum_forwards": forwards,
            "estimated_hours": hours,
            "expensive": bool(expensive_reasons),
            "expensive_reasons": expensive_reasons,
        }


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return payload


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_override(payload: dict[str, Any], override: str) -> None:
    if "=" not in override:
        raise ValueError(f"override must use KEY=VALUE syntax: {override}")
    dotted_key, raw_value = override.split("=", 1)
    keys = dotted_key.split(".")
    cursor = payload
    for key in keys[:-1]:
        value = cursor.setdefault(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"cannot assign nested override below {key}")
        cursor = value
    cursor[keys[-1]] = yaml.safe_load(raw_value)


def load_qcnn_config(
    profile: str = "laptop_8gb",
    overrides: list[str] | None = None,
    *,
    config_root: str | Path = CONFIG_ROOT,
) -> QCNNConfig:
    if profile not in PROFILE_NAMES:
        raise ValueError(f"unknown profile {profile!r}; choose from {PROFILE_NAMES}")
    root = Path(config_root)
    payload = _deep_merge(_read_yaml(root / "base.yaml"), _read_yaml(root / f"{profile}.yaml"))
    for override in overrides or []:
        _apply_override(payload, override)
    payload["profile"] = profile
    known = set(QCNNConfig.__dataclass_fields__)
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    return QCNNConfig(**payload)


def qcnn_config_from_checkpoint(payload: dict[str, Any]) -> QCNNConfig:
    """Load a recorded config with narrowly scoped legacy-key migration."""
    sanitized = dict(payload)
    sanitized.pop("checkpoint_frequency", None)
    known = set(QCNNConfig.__dataclass_fields__)
    unknown = set(sanitized) - known
    if unknown:
        raise ValueError(f"checkpoint contains unknown configuration keys: {sorted(unknown)}")
    return QCNNConfig(**sanitized)
