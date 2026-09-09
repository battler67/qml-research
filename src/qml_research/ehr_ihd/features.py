"""Train-only feature preparation and paper-inspired consensus ranking."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class FittedFeaturePipeline:
    feature_order: list[str]
    imputer: SimpleImputer
    standard_scaler: StandardScaler
    angle_scaler: MinMaxScaler

    @classmethod
    def fit(cls, frame: pd.DataFrame, feature_order: list[str]) -> FittedFeaturePipeline:
        selected = frame.loc[:, feature_order]
        imputer = SimpleImputer(strategy="median", add_indicator=False)
        values = imputer.fit_transform(selected)
        standard = StandardScaler().fit(values)
        angle = MinMaxScaler(feature_range=(0.0, np.pi), clip=True).fit(values)
        return cls(feature_order, imputer, standard, angle)

    def transform(self, frame: pd.DataFrame, *, quantum: bool = False) -> np.ndarray:
        missing = set(self.feature_order) - set(frame.columns)
        if missing:
            raise ValueError(f"input is missing selected features: {sorted(missing)}")
        values = self.imputer.transform(frame.loc[:, self.feature_order])
        scaler = self.angle_scaler if quantum else self.standard_scaler
        return np.asarray(scaler.transform(values), dtype=float)


def consensus_feature_ranking(
    frame: pd.DataFrame,
    target: np.ndarray,
    *,
    n_features: int,
    seed: int,
) -> tuple[list[str], dict[str, object]]:
    """Rank original clinical fields using three train-only selectors.

    mRMR is intentionally represented by mutual information because no maintained mRMR
    dependency is installed. The fallback is recorded in the returned evidence.
    """

    names = list(frame.columns)
    imputer = SimpleImputer(strategy="median")
    values = imputer.fit_transform(frame)
    scaled = StandardScaler().fit_transform(values)
    k = min(n_features, len(names))

    logistic = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    rfe = RFE(logistic, n_features_to_select=k, step=1).fit(scaled, target)
    rfe_rank = np.asarray(rfe.ranking_, dtype=float)

    forest = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=1
    ).fit(values, target)
    forest_order = np.argsort(-forest.feature_importances_)
    forest_rank = np.empty(len(names), dtype=float)
    forest_rank[forest_order] = np.arange(1, len(names) + 1)

    mi = partial(mutual_info_classif, random_state=seed)(values, target)
    mi_order = np.argsort(-mi)
    mi_rank = np.empty(len(names), dtype=float)
    mi_rank[mi_order] = np.arange(1, len(names) + 1)

    average = (rfe_rank + forest_rank + mi_rank) / 3.0
    final_order = np.lexsort((np.asarray(names), average))
    selected = [names[index] for index in final_order[:k]]
    evidence = {
        "method": "rank_average",
        "mrmr_status": "mutual_information_fallback_dependency_not_installed",
        "fit_scope": "training_only",
        "selected": selected,
        "rankings": {
            name: {
                "rfe_logistic": int(rfe_rank[index]),
                "random_forest": int(forest_rank[index]),
                "mutual_information_fallback": int(mi_rank[index]),
                "average_rank": float(average[index]),
            }
            for index, name in enumerate(names)
        },
    }
    return selected, evidence
