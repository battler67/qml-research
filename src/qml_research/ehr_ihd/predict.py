"""Validated research-only prediction interface for fitted UCI pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from qml_research.ehr_ihd.data import UCI_FEATURES
from qml_research.ehr_ihd.evaluation import calibrated_probabilities
from qml_research.ehr_ihd.models import decision_scores

WARNING = "Research demonstration only; not a diagnosis or treatment recommendation."
FORBIDDEN_OUTCOMES = {"num", "target", "anychd", "timechd", "mi_fchd", "timemifc"}
RANGES = {
    "age": (18, 100),
    "sex": (0, 1),
    "cp": (1, 4),
    "trestbps": (70, 260),
    "chol": (80, 700),
    "fbs": (0, 1),
    "restecg": (0, 2),
    "thalach": (50, 250),
    "exang": (0, 1),
    "oldpeak": (-3, 10),
    "slope": (1, 3),
    "ca": (0, 4),
    "thal": (0, 7),
}


def validate_uci_record(record: dict[str, Any]) -> pd.DataFrame:
    forbidden = sorted({str(key).lower() for key in record} & FORBIDDEN_OUTCOMES)
    if forbidden:
        raise ValueError(f"outcome fields are forbidden: {forbidden}")
    missing = sorted(set(UCI_FEATURES) - set(record))
    if missing:
        raise ValueError(f"required fields are missing: {missing}")
    unknown = sorted(set(record) - set(UCI_FEATURES))
    if unknown:
        raise ValueError(f"unknown fields are not accepted: {unknown}")
    clean: dict[str, float] = {}
    for field in UCI_FEATURES:
        try:
            value = float(record[field])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be numeric") from exc
        lower, upper = RANGES[field]
        if not lower <= value <= upper:
            raise ValueError(f"{field} must be between {lower} and {upper}")
        clean[field] = value
    return pd.DataFrame([clean], columns=list(UCI_FEATURES))


def predict_record(
    run_dir: str | Path, record: dict[str, Any], *, explain: bool = False
) -> dict[str, Any]:
    root = Path(run_dir)
    metadata_path = root / "models" / "prediction_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pipeline = joblib.load(root / "preprocessing" / "selected_pipeline.joblib")
    model = joblib.load(root / "models" / "prediction_model.joblib")
    calibrator = joblib.load(root / "calibration" / "prediction_calibrator.joblib")
    frame = validate_uci_record(record)
    values = pipeline.transform(frame)
    score = decision_scores(model, values)
    probability = calibrated_probabilities(calibrator, score)
    result: dict[str, Any] = {
        "prediction_task": "heart_disease_presence_classification",
        "time_horizon": None,
        "model_score": float(score[0]),
        "calibrated_probability": float(probability[0]),
        "threshold": float(metadata["threshold"]),
        "predicted_heart_disease_present": bool(probability[0] >= metadata["threshold"]),
        "model": metadata["model"],
        "run_id": metadata["run_id"],
        "dataset": "uci_heart_disease_cleveland",
        "missing_inputs": [],
        "warning": WARNING,
    }
    if explain:
        result["explanation"] = {
            "scope": "model behaviour only; not clinical causality",
            "selected_feature_values": {
                name: float(frame.iloc[0][name]) for name in pipeline.feature_order
            },
        }
    return result
