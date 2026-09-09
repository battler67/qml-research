from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qml_research.ehr_ihd.data import (
    construct_framingham_10y,
    construct_period2_landmark,
    deterministic_split,
    load_uci_cleveland,
    validate_no_future_features,
)


def test_official_uci_cache_and_binary_label() -> None:
    dataset = load_uci_cleveland()
    assert dataset.metadata["doi"] == "10.24432/C52P4X"
    assert dataset.frame.shape == (303, 13)
    assert set(np.unique(dataset.target)) == {0, 1}
    assert np.array_equal(dataset.target, (dataset.severity > 0).astype(int))
    assert dataset.frame[["ca", "thal"]].isna().any().any()


def test_deterministic_patient_level_split_is_disjoint() -> None:
    dataset = load_uci_cleveland()
    first = deterministic_split(dataset, 42)
    second = deterministic_split(dataset, 42)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert sum(map(len, first.values())) == 303


def test_framingham_prevalence_censoring_and_horizon() -> None:
    raw = pd.DataFrame(
        {
            "RANDID": [1, 2, 3, 4, 5, 1],
            "PERIOD": [1, 1, 1, 1, 1, 2],
            "PREVCHD": [0, 0, 0, 1, 0, 0],
            "ANYCHD": [1, 0, 0, 1, 1, 1],
            "TIMECHD": [1000, 5000, 1000, 100, 5000, 1000],
            "AGE": [40, 50, 60, 70, 45, 46],
        }
    )
    cohort, target, report = construct_framingham_10y(raw)
    assert cohort["RANDID"].tolist() == [1, 2, 5]
    assert target.tolist() == [1, 0, 0]
    assert report["excluded_prevalent_chd"] == 1
    assert report["censored_or_unknown"] == 1


def test_framingham_duplicate_baseline_participant_rejected() -> None:
    raw = pd.DataFrame(
        {
            "RANDID": [1, 1],
            "PERIOD": [1, 1],
            "PREVCHD": [0, 0],
            "ANYCHD": [0, 0],
            "TIMECHD": [5000, 5000],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        construct_framingham_10y(raw)


def test_future_feature_names_are_rejected() -> None:
    validate_no_future_features(["AGE", "SYSBP"])
    with pytest.raises(ValueError, match="TIMECHD"):
        validate_no_future_features(["AGE", "TIMECHD"])


def test_period2_landmark_uses_relative_time_and_prior_measurements_only() -> None:
    raw = pd.DataFrame(
        {
            "RANDID": [1, 1, 2, 2, 3, 3],
            "PERIOD": [1, 2, 1, 2, 1, 2],
            "PREVCHD": [0, 0, 0, 0, 0, 1],
            "TIME": [0, 2000, 0, 2100, 0, 1900],
            "ANYCHD": [1, 1, 0, 0, 1, 1],
            "TIMECHD": [3000, 3000, 7000, 7000, 2500, 2500],
            "SYSBP": [120, 135, 130, 128, 140, 145],
        }
    )
    frame, target, report = construct_period2_landmark(raw, ["SYSBP"])
    assert frame["RANDID"].tolist() == [1, 2]
    assert frame["relative_followup_days"].tolist() == [1000, 4900]
    assert frame["sysbp_absolute_change"].tolist() == [15, -2]
    assert target.tolist() == [1, 0]
    assert report["excluded_prevalent_chd_period2"] == 1
