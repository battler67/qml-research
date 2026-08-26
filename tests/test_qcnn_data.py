from dataclasses import replace

import numpy as np

from qml_research.qcnn.config import load_qcnn_config
from qml_research.qcnn.data import FeatureReducer, load_wdbc_fold


def test_spatial_pool_preserves_shape_and_angle_range() -> None:
    images = np.arange(6 * 28 * 28, dtype=np.uint8).reshape(6, 28, 28)
    reducer = FeatureReducer("spatial_pool", 4, 7, is_image=True)
    reduced = reducer.fit_transform(images)
    assert reduced.shape == (6, 4)
    assert np.all(reduced >= -np.pi)
    assert np.all(reduced <= np.pi)
    assert reducer.fit_sample_count == 6


def test_pca_state_is_fitted_on_training_samples_only() -> None:
    rng = np.random.default_rng(4)
    train = rng.normal(size=(20, 12))
    test = rng.normal(loc=100, size=(5, 12))
    reducer = FeatureReducer("pca", 4, 7, is_image=False)
    reducer.fit(train)
    transformed = reducer.transform(test)
    assert transformed.shape == (5, 4)
    assert reducer.fit_sample_count == 20
    assert len(reducer.metadata()["explained_variance_ratio"]) == 4


def test_wdbc_fold_is_disjoint_stratified_and_disease_positive() -> None:
    config = replace(
        load_qcnn_config("smoke"),
        dataset="wdbc",
        reducer="pca",
        train_size=32,
        validation_size=16,
        test_size=32,
    )
    splits = load_wdbc_fold(config, seed=7, fold=0)
    assert len(splits.train.y) == 32
    assert len(splits.validation.y) == 16
    assert len(splits.test.y) == 32
    assert set(splits.train.indices).isdisjoint(splits.validation.indices)
    assert set(splits.train.indices).isdisjoint(splits.test.indices)
    assert set(np.unique(splits.train.y)) == {0, 1}
    assert "1=malignant" in splits.provenance["target_mapping"]
