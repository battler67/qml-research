from __future__ import annotations

import pytest

from qml_research.ehr_ihd.predict import validate_uci_record

VALID = {
    "age": 58,
    "sex": 1,
    "cp": 2,
    "trestbps": 140,
    "chol": 240,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.2,
    "slope": 1,
    "ca": 0,
    "thal": 2,
}


def test_prediction_record_feature_order() -> None:
    frame = validate_uci_record(dict(reversed(list(VALID.items()))))
    assert frame.columns.tolist()[0] == "age"
    assert frame.columns.tolist()[-1] == "thal"


def test_prediction_rejects_outcome_missing_and_range() -> None:
    with pytest.raises(ValueError, match="outcome"):
        validate_uci_record({**VALID, "num": 1})
    missing = dict(VALID)
    missing.pop("age")
    with pytest.raises(ValueError, match="missing"):
        validate_uci_record(missing)
    with pytest.raises(ValueError, match="between"):
        validate_uci_record({**VALID, "age": 200})
