"""Official-data ingestion and leakage-safe endpoint construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from qml_research.ehr_ihd.config import PROJECT_ROOT

UCI_FEATURES = (
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
)
UCI_CATEGORICAL = ("sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")
FRAMINGHAM_LEAKAGE_FIELDS = {
    "ANYCHD",
    "ANGINA",
    "HOSPMI",
    "MI_FCHD",
    "CVD",
    "DEATH",
    "TIMECHD",
    "TIMEAP",
    "TIMEMI",
    "TIMEMIFC",
    "TIMECVD",
    "TIMEDTH",
    "GLUCOSEYEAR6",
    "NHOSP",
}


@dataclass(frozen=True)
class IHDDataset:
    frame: pd.DataFrame
    target: np.ndarray
    severity: np.ndarray | None
    patient_ids: np.ndarray
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_uci_cleveland(cache_dir: str | Path = "data/raw") -> IHDDataset:
    """Load the cached official UCI Cleveland cohort, fetching via existing loader if absent."""

    from qml_research.data.loaders import load_heart

    root = PROJECT_ROOT / cache_dir
    bundle = load_heart(root)
    cache = root / "uci_heart_disease.csv"
    raw = pd.read_csv(cache, na_values="?")
    target_name = "target" if "target" in raw else "num"
    severity = pd.to_numeric(raw[target_name], errors="coerce")
    valid = severity.notna()
    frame = raw.loc[valid, list(UCI_FEATURES)].apply(pd.to_numeric, errors="coerce")
    frame = frame.reset_index(drop=True)
    severity_values = severity.loc[valid].astype(int).to_numpy()
    target = (severity_values > 0).astype(int)
    metadata = {
        **bundle.provenance,
        "dataset": "UCI Heart Disease",
        "cohort": "Cleveland",
        "doi": "10.24432/C52P4X",
        "url": "https://archive.ics.uci.edu/dataset/45/heart+disease",
        "cache_sha256": _sha256(cache),
        "raw_rows": int(len(raw)),
        "effective_rows": int(len(frame)),
        "binary_mapping": "num == 0: absent; num in 1..4: present",
        "prediction_type": "heart-disease presence classification",
    }
    return IHDDataset(
        frame=frame,
        target=target,
        severity=severity_values,
        patient_ids=np.arange(len(frame), dtype=int),
        metadata=metadata,
    )


def validate_no_future_features(columns: list[str] | tuple[str, ...]) -> None:
    normalized = {str(column).upper() for column in columns}
    leaked = sorted(normalized & FRAMINGHAM_LEAKAGE_FIELDS)
    if leaked:
        raise ValueError(f"future/outcome leakage fields are forbidden: {leaked}")


def construct_framingham_10y(
    raw: pd.DataFrame,
    *,
    horizon_days: float = 3652.5,
    endpoint: str = "ANYCHD",
    event_time: str = "TIMECHD",
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """Construct the documented baseline incident-CHD binary cohort.

    TIMECHD/TIMEMIFC is treated as event time for cases and observed follow-up time for
    non-cases, as specified by the teaching data documentation.
    """

    frame = raw.copy()
    frame.columns = [str(column).upper() for column in frame.columns]
    required = {"RANDID", "PERIOD", "PREVCHD", endpoint, event_time}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Framingham data is missing required columns: {sorted(missing)}")
    baseline = frame.loc[frame["PERIOD"] == 1].copy()
    duplicate_ids = int(baseline["RANDID"].duplicated().sum())
    if duplicate_ids:
        raise ValueError("Framingham baseline contains duplicate participant IDs")
    eligible = baseline.loc[baseline["PREVCHD"] == 0].copy()
    status = pd.to_numeric(eligible[endpoint], errors="coerce")
    time = pd.to_numeric(eligible[event_time], errors="coerce")
    positive = (status == 1) & (time <= horizon_days)
    negative = ((status == 0) | ((status == 1) & (time > horizon_days))) & (time >= horizon_days)
    known = positive | negative
    cohort = eligible.loc[known].copy().reset_index(drop=True)
    target = positive.loc[known].astype(int).to_numpy()
    report = {
        "raw_rows": int(len(frame)),
        "baseline_participants": int(len(baseline)),
        "excluded_prevalent_chd": int((baseline["PREVCHD"] != 0).sum()),
        "eligible_disease_free": int(len(eligible)),
        "positive_within_10y": int(positive.sum()),
        "negative_adequate_followup": int(negative.sum()),
        "censored_or_unknown": int((~known).sum()),
        "binary_cohort": int(known.sum()),
    }
    return cohort, target, report


def load_framingham_teaching(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Official Framingham teaching data not found at {source}. See docs/data_access.md"
        )
    frame = pd.read_csv(source)
    frame.columns = [str(column).upper() for column in frame.columns]
    return frame


def construct_period2_landmark(
    raw: pd.DataFrame,
    feature_columns: list[str],
    *,
    horizon_days: float = 3652.5,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """Build leakage-safe Period-2 longitudinal features and a relative 10-year endpoint."""

    frame = raw.copy()
    frame.columns = [str(column).upper() for column in frame.columns]
    features = [column.upper() for column in feature_columns]
    validate_no_future_features(features)
    required = {"RANDID", "PERIOD", "PREVCHD", "TIME", "ANYCHD", "TIMECHD", *features}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Framingham landmark data is missing columns: {sorted(missing)}")
    first = frame.loc[frame["PERIOD"] == 1].set_index("RANDID")
    second = frame.loc[frame["PERIOD"] == 2].set_index("RANDID")
    common = first.index.intersection(second.index)
    first = first.loc[common]
    second = second.loc[common]
    eligible = pd.to_numeric(second["PREVCHD"], errors="coerce") == 0
    first = first.loc[eligible]
    second = second.loc[eligible]
    index_time = pd.to_numeric(second["TIME"], errors="coerce")
    relative_time = pd.to_numeric(second["TIMECHD"], errors="coerce") - index_time
    status = pd.to_numeric(second["ANYCHD"], errors="coerce")
    positive = (status == 1) & (relative_time > 0) & (relative_time <= horizon_days)
    negative = ((status == 0) | ((status == 1) & (relative_time > horizon_days))) & (
        relative_time >= horizon_days
    )
    known = positive | negative
    output: dict[str, pd.Series] = {}
    years = (index_time / 365.25).replace(0, np.nan)
    for feature in features:
        initial = pd.to_numeric(first[feature], errors="coerce")
        current = pd.to_numeric(second[feature], errors="coerce")
        delta = current - initial
        denominator = initial.abs().replace(0, np.nan)
        output[f"{feature.lower()}_period1"] = initial
        output[f"{feature.lower()}_period2"] = current
        output[f"{feature.lower()}_absolute_change"] = delta
        output[f"{feature.lower()}_percentage_change"] = delta / denominator
        output[f"{feature.lower()}_slope_per_year"] = delta / years
    longitudinal = pd.DataFrame(output, index=second.index).loc[known].reset_index()
    longitudinal["relative_followup_days"] = relative_time.loc[known].to_numpy()
    report = {
        "participants_with_period1_and_period2": int(len(common)),
        "excluded_prevalent_chd_period2": int((~eligible).sum()),
        "eligible_period2": int(eligible.sum()),
        "positive_within_10y": int(positive.sum()),
        "negative_adequate_followup": int(negative.sum()),
        "censored_or_unknown": int((~known).sum()),
    }
    return longitudinal, positive.loc[known].astype(int).to_numpy(), report


def deterministic_split(dataset: IHDDataset, seed: int = 42) -> dict[str, np.ndarray]:
    indices = np.arange(len(dataset.target))
    train, remainder = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=dataset.target
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=seed,
        stratify=dataset.target[remainder],
    )
    return {
        "train": np.sort(train),
        "validation": np.sort(validation),
        "test": np.sort(test),
    }


def dataset_report(dataset: IHDDataset, split: dict[str, np.ndarray]) -> dict[str, Any]:
    frame = dataset.frame
    return {
        "rows": int(len(frame)),
        "features": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_patient_ids": int(pd.Series(dataset.patient_ids).duplicated().sum()),
        "missingness": {key: int(value) for key, value in frame.isna().sum().items()},
        "numeric_ranges": {
            column: {
                "min": None if frame[column].dropna().empty else float(frame[column].min()),
                "max": None if frame[column].dropna().empty else float(frame[column].max()),
            }
            for column in frame.columns
        },
        "class_counts": {
            str(label): int(count)
            for label, count in zip(*np.unique(dataset.target, return_counts=True), strict=True)
        },
        "split_counts": {
            name: {
                "rows": int(len(indices)),
                "positive": int(dataset.target[indices].sum()),
            }
            for name, indices in split.items()
        },
    }


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
