"""Validated configuration for the isolated EHR IHD experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = PROJECT_ROOT / "experiments" / "ehr_ihd_qml" / "configs"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    return payload


@dataclass(frozen=True)
class IHDConfig:
    """Small validated wrapper that preserves the resolved YAML contract."""

    payload: dict[str, Any]
    source: Path

    def section(self, name: str) -> dict[str, Any]:
        value = self.payload.get(name, {})
        if not isinstance(value, dict):
            raise ValueError(f"configuration section {name!r} must be a mapping")
        return value

    @property
    def track(self) -> str:
        return str(self.section("experiment").get("track", "uci_diagnostic"))

    @property
    def seed(self) -> int:
        return int(self.section("experiment").get("seed", 42))

    @property
    def output_dir(self) -> Path:
        return PROJECT_ROOT / str(
            self.section("experiment").get("output_dir", "results/ehr_ihd_qml")
        )

    def fingerprint(self) -> str:
        encoded = json.dumps(self.payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]

    def validate(self) -> None:
        if self.track not in {"uci_diagnostic", "framingham_early_prediction", "paper_replication"}:
            raise ValueError(f"unsupported experiment track: {self.track}")
        quantum = self.section("quantum")
        features = int(
            quantum.get("features", self.section("feature_selection").get("n_features", 4))
        )
        if features not in {4, 6, 8, 10}:
            raise ValueError("quantum features must be one of 4, 6, 8, or 10")
        if int(quantum.get("layers", 1)) not in {1, 2, 3}:
            raise ValueError("quantum layers must be between 1 and 3")
        if quantum.get("backend", "statevector") != "statevector":
            raise ValueError("this local experiment currently supports statevector only")
        if quantum.get("real_hardware_submit", False):
            raise ValueError("real hardware submission is intentionally disabled")
        resources = self.section("resources")
        profile = str(self.section("experiment").get("profile", "laptop_8gb"))
        memory_ceiling = 20.0 if profile == "workstation_24gb" else 6.5
        if float(resources.get("memory_limit_gb", 5.5)) > memory_ceiling:
            raise ValueError(f"memory_limit_gb exceeds the safe ceiling for {profile}")
        if int(resources.get("max_kernel_evaluations", 20_000)) < 1:
            raise ValueError("max_kernel_evaluations must be positive")
        loss = self.section("qmlp").get("loss", "bce_logits")
        if loss not in {"bce_logits", "mse_paper_ablation"}:
            raise ValueError("unsupported QMLP loss")
        entanglement = self.section("qmlp").get("entanglement", "linear")
        if entanglement not in {"linear", "circular"}:
            raise ValueError("QMLP entanglement must be linear or circular")


def load_ihd_config(path: str | Path) -> IHDConfig:
    source = Path(path).resolve()
    config = IHDConfig(payload=_read_yaml(source), source=source)
    config.validate()
    return config
