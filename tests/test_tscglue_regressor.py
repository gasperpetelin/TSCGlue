"""Tests for TSCGlueRegressor."""

import tempfile

import numpy as np

from tscglue.models_regressor import TSCGlueRegressor


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
