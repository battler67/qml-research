from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qml_research.ehr_ihd.config import IHDConfig, load_ihd_config
from qml_research.ehr_ihd.features import FittedFeaturePipeline, consensus_feature_ranking
from qml_research.ehr_ihd.runner import _run_directory

CONFIG_ROOT = Path("experiments/ehr_ihd_qml/configs")


@pytest.mark.parametrize(
    "name",
    [
        "uci_smoke.yaml",
        "framingham_early_chd.yaml",
        "paper_replication.yaml",
        "laptop_8gb.yaml",
        "workstation_24gb.yaml",
        "tuning.yaml",
    ],
)
def test_required_configs_parse(name: str) -> None:
    assert load_ihd_config(CONFIG_ROOT / name).payload


def test_incompatible_quantum_feature_count_rejected(tmp_path: Path) -> None:
    config = IHDConfig(
        payload={
            "experiment": {"track": "uci_diagnostic"},
            "quantum": {"features": 5, "layers": 1, "backend": "statevector"},
        },
        source=tmp_path / "invalid.yaml",
    )
    with pytest.raises(ValueError, match="4, 6, 8, or 10"):
        config.validate()


def test_consensus_and_pipeline_preserve_feature_order() -> None:
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(rng.normal(size=(60, 10)), columns=[f"f{i}" for i in range(10)])
    frame.loc[0, "f2"] = np.nan
    target = (frame["f0"].fillna(0) + frame["f1"] > 0).astype(int).to_numpy()
    selected, evidence = consensus_feature_ranking(frame, target, n_features=4, seed=42)
    pipeline = FittedFeaturePipeline.fit(frame.iloc[:40], selected)
    transformed = pipeline.transform(frame.iloc[40:])
    quantum = pipeline.transform(frame.iloc[40:], quantum=True)
    assert len(selected) == 4
    assert pipeline.feature_order == selected
    assert transformed.shape == (20, 4)
    assert quantum.min() >= 0 and quantum.max() <= np.pi
    assert evidence["fit_scope"] == "training_only"


def test_result_directory_never_overwrites(tmp_path: Path) -> None:
    config = IHDConfig(
        payload={
            "experiment": {
                "track": "uci_diagnostic",
                "output_dir": str(tmp_path),
            },
            "quantum": {"features": 4, "layers": 1, "backend": "statevector"},
            "resources": {"memory_limit_gb": 1, "max_kernel_evaluations": 10},
            "qmlp": {"loss": "bce_logits", "entanglement": "linear"},
        },
        source=tmp_path / "test.yaml",
    )
    _, created = _run_directory(config, "fixed-run")
    assert created.exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        _run_directory(config, "fixed-run")
