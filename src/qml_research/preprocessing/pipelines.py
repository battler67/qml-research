"""Fold-local preprocessing shared by matched classical and quantum models."""

from __future__ import annotations

from functools import partial

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from qml_research.data import DatasetBundle


def _column_transformer(bundle: DatasetBundle) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if bundle.numerical_columns:
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric, list(bundle.numerical_columns)))
    if bundle.categorical_columns:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=float),
                ),
            ]
        )
        transformers.append(("categorical", categorical, list(bundle.categorical_columns)))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def make_full_preprocessor(bundle: DatasetBundle) -> Pipeline:
    """Preprocess all features for the full-feature classical reference."""

    return Pipeline([("columns", _column_transformer(bundle))])


def make_preprocessor(
    bundle: DatasetBundle,
    method: str,
    n_features: int,
    seed: int,
) -> Pipeline:
    """Build a fold-fitted reduction and angle-scaling pipeline.

    The returned object is intentionally unfitted. Callers must fit it on an outer
    training fold and use only ``transform`` for validation/test data.
    """

    if n_features < 1:
        raise ValueError("n_features must be positive")
    if method == "pca":
        reducer = PCA(n_components=n_features, random_state=seed)
    elif method == "mutual_info":
        reducer = SelectKBest(
            score_func=partial(mutual_info_classif, random_state=seed), k=n_features
        )
    else:
        raise ValueError(f"unsupported reduction method: {method}")

    return Pipeline(
        [
            ("columns", _column_transformer(bundle)),
            ("reduce", reducer),
            ("angle_scale", MinMaxScaler(feature_range=(-np.pi, np.pi), clip=True)),
        ]
    )
