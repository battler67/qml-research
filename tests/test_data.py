import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from qml_research.data import load_dataset


def test_iris_binary_loader() -> None:
    bundle = load_dataset("iris")
    assert bundle.X.shape == (100, 4)
    assert np.bincount(bundle.y).tolist() == [50, 50]
    assert "Non-medical" in bundle.task_description


def test_wdbc_maps_malignant_to_positive() -> None:
    bundle = load_dataset("wdbc")
    assert bundle.X.shape == (569, 30)
    assert np.bincount(bundle.y).tolist() == [357, 212]
    assert bundle.positive_class.startswith("Malignant")
    assert "not longitudinal" in bundle.task_description


def test_heart_cache_provenance_and_disease_presence_mapping(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "age": [40, 50, 60, 70],
            "sex": [0, 1, 1, 0],
            "cp": [1, 2, 3, 4],
            "trestbps": [120, 130, 140, 150],
            "chol": [180, 200, 220, 240],
            "fbs": [0, 0, 1, 1],
            "restecg": [0, 1, 2, 0],
            "thalach": [170, 160, 150, 140],
            "exang": [0, 0, 1, 1],
            "oldpeak": [0.0, 1.0, 2.0, 3.0],
            "slope": [1, 2, 2, 3],
            "ca": [0, 0, 1, 2],
            "thal": [3, 3, 6, 7],
            "target": [0, 1, 2, 4],
        }
    )
    cache = tmp_path / "uci_heart_disease.csv"
    manifest = tmp_path / "uci_heart_disease.provenance.json"
    frame.to_csv(cache, index=False)
    checksum = hashlib.sha256(cache.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps(
            {
                "source": "unit fixture for official UCI schema",
                "sha256": checksum,
                "public_non_identifiable": True,
            }
        ),
        encoding="utf-8",
    )

    bundle = load_dataset("heart", tmp_path)

    assert bundle.X.shape == (4, 13)
    assert bundle.y.tolist() == [0, 1, 1, 1]
    assert bundle.provenance["sha256"] == checksum
    assert bundle.provenance["target_mapping"] == "0=no disease, values 1-4=disease present"
