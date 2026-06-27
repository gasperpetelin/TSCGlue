"""Tests for TSCGlueClassifier, TSCGlueRegressor, and LokyStackerV10Base."""

import tempfile

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from tscglue import utils
from tscglue.models import (
    LokyStackerV10Base,
    TSCAGGlueClassifier,
    TSCGlueClassifier,
    TSCGlueRegressor,
)


def test_model_accuracy_on_coffee():
    X_train, y_train, X_test, y_test = utils.load_dataset("Coffee")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(random_state=270, n_repetitions=1, k_folds=10, runs_dir=tmp_dir)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


def test_ag_stacking_on_coffee():
    X_train, y_train, X_test, y_test = utils.load_dataset("Coffee")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCAGGlueClassifier(random_state=270, n_repetitions=1, k_folds=10, runs_dir=tmp_dir)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


def _make_classification_data(n_per_class=10, n_classes=15, n_timesteps=16, seed=0):
    """Synthetic multiclass series with INTEGER labels 0..n_classes-1.

    15 integer-labeled classes is what exercises the multiclass roc_auc
    label-ordering path (labels 10..14 sort differently by repr vs numerically).
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        center = rng.standard_normal(n_timesteps) + c
        for _ in range(n_per_class):
            X.append((center + 0.3 * rng.standard_normal(n_timesteps))[None, :])
            y.append(c)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int)


@pytest.mark.parametrize("eval_metric", ["accuracy", "log_loss", "roc_auc"])
def test_classifier_eval_metrics_multiclass(eval_metric):
    # 15 integer-labeled classes -> roc_auc would raise "labels must be ordered"
    # before the label-sorting fix.
    X_train, y_train = _make_classification_data(seed=0)
    X_test, y_test = _make_classification_data(seed=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0,
            n_repetitions=1,
            k_folds=3,
            n_jobs=2,
            eval_metric=eval_metric,
            runs_dir=tmp_dir,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

    assert y_pred.shape == (len(X_test),)
    assert set(np.unique(y_pred)).issubset(set(np.unique(y_train)))
    assert proba.shape == (len(X_test), len(np.unique(y_train)))
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("feature_dtype", [None, "float32", "float64"])
def test_v10base_feature_dtype(feature_dtype):
    """Test that LokyStackerV10Base fit+predict works with different feature_dtype values."""
    X_train, y_train, X_test, y_test = utils.load_dataset("Coffee")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = LokyStackerV10Base(
            random_state=270,
            n_repetitions=1,
            k_folds=10,
            feature_dtype=feature_dtype,
            runs_dir=tmp_dir,
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


def test_model_on_multivariate():
    X_train, y_train, X_test, y_test = utils.load_dataset("BasicMotions")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(random_state=270, n_repetitions=1, k_folds=10, runs_dir=tmp_dir)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


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

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=270, n_repetitions=1, k_folds=10, n_jobs=1, runs_dir=tmp_dir
        )
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

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueRegressor(random_state=0, k_folds=3, n_jobs=1, runs_dir=tmp_dir)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    assert y_pred.shape == (len(X_test),), f"Expected shape ({len(X_test)},), got {y_pred.shape}"
    assert np.isfinite(y_pred).all(), "Predictions contain NaN or Inf"
    assert y_pred.dtype in (np.float32, np.float64), f"Unexpected dtype {y_pred.dtype}"


def test_regressor_summary():
    X_train, y_train, X_test, _ = _make_regression_data()

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueRegressor(random_state=0, k_folds=3, n_jobs=1, runs_dir=tmp_dir)
        model.fit(X_train, y_train)
        scores = model.summary()
        scores_with_transforms = model.summary(return_transforms=True)

    assert len(scores) > 0
    for entry in scores:
        assert "model" in entry
        assert "level" in entry
        assert "oof_rmse" in entry
        assert "oof_r2" in entry
        assert "train_time" in entry
        assert np.isfinite(entry["oof_rmse"]), f"oof_rmse is not finite for {entry['model']}"
        assert np.isfinite(entry["oof_r2"]), f"oof_r2 is not finite for {entry['model']}"

    assert len(scores_with_transforms) >= len(scores)
