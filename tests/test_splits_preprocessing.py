import numpy as np

from qml_research.data import load_dataset, make_stratified_folds, stratified_subsample
from qml_research.preprocessing import make_preprocessor


def test_stratified_sampling_and_folds_are_reproducible() -> None:
    bundle = load_dataset("wdbc")
    first = stratified_subsample(bundle.y, 100, 42)
    second = stratified_subsample(bundle.y, 100, 42)
    assert np.array_equal(first, second)
    folds_a = make_stratified_folds(bundle.y[first], 5, 42)
    folds_b = make_stratified_folds(bundle.y[first], 5, 42)
    assert all(
        np.array_equal(train_a, train_b) and np.array_equal(test_a, test_b)
        for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b, strict=True)
    )


def test_preprocessing_fits_only_training_rows() -> None:
    bundle = load_dataset("iris")
    train = bundle.X.iloc[:80].copy()
    test = bundle.X.iloc[80:].copy()
    test.iloc[:, :] = 1_000_000
    pipeline = make_preprocessor(bundle, "pca", 2, 42)
    transformed_train = pipeline.fit_transform(train, bundle.y[:80])
    transformed_test = pipeline.transform(test)
    scaler = pipeline.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
    assert np.max(scaler.mean_) < 10
    assert transformed_train.shape == (80, 2)
    assert transformed_test.shape == (20, 2)
    assert np.max(transformed_test) <= np.pi


def test_mutual_information_pipeline_is_seeded() -> None:
    bundle = load_dataset("wdbc")
    pipeline_a = make_preprocessor(bundle, "mutual_info", 4, 123)
    pipeline_b = make_preprocessor(bundle, "mutual_info", 4, 123)
    output_a = pipeline_a.fit_transform(bundle.X.iloc[:100], bundle.y[:100])
    output_b = pipeline_b.fit_transform(bundle.X.iloc[:100], bundle.y[:100])
    assert np.allclose(output_a, output_b)
