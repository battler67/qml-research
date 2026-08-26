"""Validated experiment configuration and stable hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DATASETS = {"iris", "wdbc", "heart"}
MODELS = {
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "qksvm",
    "vqc",
}
REDUCTION_METHODS = {"pca", "mutual_info"}
FEATURE_MAPS = {"angle", "iqp"}
QUANTUM_CONDITIONS = {"ideal", "finite_shot", "depolarizing"}


@dataclass(frozen=True)
class ExperimentConfig:
    """Decision-complete experiment matrix loaded from YAML."""

    name: str
    dataset: str
    sample_sizes: tuple[int | str, ...] = ("full",)
    seeds: tuple[int, ...] = (42, 123, 2026)
    folds: int = 5
    fold_limit: int | None = None
    feature_counts: tuple[int, ...] = (4,)
    reduction_methods: tuple[str, ...] = ("pca",)
    models: tuple[str, ...] = tuple(sorted(MODELS))
    feature_maps: tuple[str, ...] = ("angle",)
    vqc_layers: tuple[int, ...] = (1,)
    vqc_steps: int = 100
    vqc_batch_size: int = 16
    optimizer: str = "adam"
    learning_rate: float = 0.05
    backend: str = "default.qubit"
    shots: int | None = None
    finite_shots: int = 1024
    noise_strength: float = 0.0
    quantum_conditions: tuple[str, ...] = ("ideal",)
    include_full_classical: bool = True
    max_runtime_hours: float = 8.0
    output_dir: str = "results/raw"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if self.dataset not in DATASETS:
            raise ValueError(f"unsupported dataset: {self.dataset}")
        unknown_models = set(self.models) - MODELS
        if unknown_models:
            raise ValueError(f"unsupported models: {sorted(unknown_models)}")
        unknown_reducers = set(self.reduction_methods) - REDUCTION_METHODS
        if unknown_reducers:
            raise ValueError(f"unsupported reduction methods: {sorted(unknown_reducers)}")
        unknown_maps = set(self.feature_maps) - FEATURE_MAPS
        if unknown_maps:
            raise ValueError(f"unsupported feature maps: {sorted(unknown_maps)}")
        unknown_conditions = set(self.quantum_conditions) - QUANTUM_CONDITIONS
        if unknown_conditions:
            raise ValueError(f"unsupported quantum conditions: {sorted(unknown_conditions)}")
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        if self.fold_limit is not None and not 1 <= self.fold_limit <= self.folds:
            raise ValueError("fold_limit must be between 1 and folds")
        if not self.seeds or not self.feature_counts:
            raise ValueError("seeds and feature_counts must not be empty")
        if any(count < 1 or count > 8 for count in self.feature_counts):
            raise ValueError("feature counts must be between 1 and 8")
        if any(layer not in {1, 2} for layer in self.vqc_layers):
            raise ValueError("Phase 1 VQC layers must be 1 or 2")
        if self.vqc_steps < 1 or self.vqc_batch_size < 1:
            raise ValueError("VQC steps and batch size must be positive")
        if self.optimizer != "adam":
            raise ValueError("Phase 1 supports the adam optimizer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.backend != "default.qubit":
            raise ValueError("Phase 1 supports default.qubit; noisy kernels use default.mixed")
        if self.shots is not None and self.shots < 1:
            raise ValueError("shots must be positive or null")
        if self.finite_shots < 1:
            raise ValueError("finite_shots must be positive")
        if not 0.0 <= self.noise_strength < 1.0:
            raise ValueError("noise_strength must be in [0, 1)")
        for size in self.sample_sizes:
            if size != "full" and (not isinstance(size, int) or size < self.folds * 2):
                raise ValueError("sample sizes must be 'full' or support stratified folds")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self, extra: dict[str, Any] | None = None) -> str:
        payload = self.to_dict()
        if extra:
            payload["run"] = extra
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _tuplify(data: dict[str, Any]) -> dict[str, Any]:
    tuple_fields = {
        "sample_sizes",
        "seeds",
        "feature_counts",
        "reduction_methods",
        "models",
        "feature_maps",
        "vqc_layers",
        "quantum_conditions",
    }
    return {key: tuple(value) if key in tuple_fields else value for key, value in data.items()}


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"configuration must be a YAML mapping: {config_path}")
    return ExperimentConfig(**_tuplify(data))
