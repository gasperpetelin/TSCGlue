"""Tests for TSCGlueClassifier, TSCGlueRegressor, and LokyStackerV10Base."""

import numpy as np
import pytest
from aeon.datasets import load_regression
from sklearn.metrics import accuracy_score
from tscglue.models import LokyStackerV10Base, TSCGlueClassifier, TSCGlueRegressor
from tscglue import utils


def test_model_accuracy_on_coffee():
    X_train, y_train, X_test, y_test = utils.load_dataset("Coffee")

    model = TSCGlueClassifier(random_state=270, n_repetitions=1, k_folds=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


@pytest.mark.parametrize("feature_dtype", [None, "float32", "float64"])
def test_v10base_feature_dtype(feature_dtype):
    """Test that LokyStackerV10Base fit+predict works with different feature_dtype values."""
    X_train, y_train, X_test, y_test = utils.load_dataset("Coffee")

    model = LokyStackerV10Base(
        random_state=270, n_repetitions=1, k_folds=10, feature_dtype=feature_dtype,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"

    expected_dtype = np.dtype(feature_dtype) if feature_dtype else X_train.dtype
    assert model.feature_dtype == expected_dtype, (
        f"Expected feature_dtype={expected_dtype}, got {model.feature_dtype}"
    )


@pytest.mark.skip(reason="TSCGlueClassifier foundation models (Chronos2) don't support multivariate yet")
def test_model_on_multivariate():
    X_train, y_train, X_test, y_test = utils.load_dataset("BasicMotions")

    model = TSCGlueClassifier(random_state=270, n_repetitions=1, k_folds=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


@pytest.mark.skip(reason="TSCGlueClassifier foundation models (Chronos2) don't support multivariate yet")
@pytest.mark.parametrize("encode_labels", [False, True], ids=["string_labels", "int_labels"])
def test_label_dtype(encode_labels):
    """Test that inference works with both string and integer labels."""
    X_train, y_train, X_test, y_test = utils.load_dataset("BasicMotions")

    if encode_labels:
        labels, y_train_fit = np.unique(y_train, return_inverse=True)
        y_test_expected = np.array([np.where(labels == x)[0][0] for x in y_test])
    else:
        y_train_fit = y_train
        y_test_expected = y_test

    model = TSCGlueClassifier(random_state=270, n_repetitions=1, k_folds=10, n_jobs=1)
    model.fit(X_train, y_train_fit)

    y_pred = model.predict(X_test)
    proba_per_model = model.predict_proba_per_model(X_test)
    best_proba = proba_per_model[model.best_model]

    assert y_pred.shape == y_test_expected.shape
    assert best_proba.shape == (X_test.shape[0], len(model.classes_))
    assert np.isfinite(best_proba).all()

    accuracy = accuracy_score(y_test_expected, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


def _make_regression_data(n_train=40, n_test=15, n_channels=1, n_timesteps=30, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal((n_train, n_channels, n_timesteps)).astype(np.float32)
    X_test = rng.standard_normal((n_test, n_channels, n_timesteps)).astype(np.float32)
    y_train = X_train[:, 0, :].mean(axis=1) + 0.1 * rng.standard_normal(n_train)
    y_test = X_test[:, 0, :].mean(axis=1) + 0.1 * rng.standard_normal(n_test)
    return X_train, y_train, X_test, y_test


def test_regressor_fit_predict_basic():
    X_train, y_train, X_test, y_test = _make_regression_data()
    model = TSCGlueRegressor(random_state=0, k_folds=3, n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred.shape == (len(X_test),), f"Expected shape ({len(X_test)},), got {y_pred.shape}"
    assert np.isfinite(y_pred).all(), "Predictions contain NaN or Inf"
    assert y_pred.dtype in (np.float32, np.float64), f"Unexpected dtype {y_pred.dtype}"


def test_regressor_summary():
    X_train, y_train, X_test, _ = _make_regression_data()
    model = TSCGlueRegressor(random_state=0, k_folds=3, n_jobs=1)
    model.fit(X_train, y_train)

    scores = model.summary()
    assert len(scores) > 0
    for entry in scores:
        assert "model" in entry
        assert "level" in entry
        assert "oof_rmse" in entry
        assert "oof_r2" in entry
        assert "train_time" in entry
        assert np.isfinite(entry["oof_rmse"]), f"oof_rmse is not finite for {entry['model']}"
        assert np.isfinite(entry["oof_r2"]), f"oof_r2 is not finite for {entry['model']}"

    scores_with_transforms = model.summary(return_transforms=True)
    assert len(scores_with_transforms) >= len(scores)


def _normalize(X):
    mean = X.mean(axis=-1, keepdims=True)
    std = X.std(axis=-1, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (X - mean) / std


def test_regressor_univariate():
    """Test regressor on real univariate regression dataset (Covid3Month, 1 channel)."""
    X_train, y_train = load_regression("Covid3Month", split="train")
    X_test, y_test = load_regression("Covid3Month", split="test")

    X_train = _normalize(X_train)
    X_test = _normalize(X_test)

    model = TSCGlueRegressor(random_state=0, k_folds=3, n_jobs=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert y_pred.shape == (len(X_test),)
    assert np.isfinite(y_pred).all()
