import numpy as np
import pytest
from sklearn.base import clone
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_info

from tscglue.tabular import (
    AutoSelectKBestClassifier,
    ClippedRegressor,
    MLPClassifierThreadLimited,
    NoScaler,
    RareClassSafeLogisticCV,
    RidgeClassifierCVDecisionProba,
    RidgeClassifierCVIndicator,
    SparseScaler,
)

HEADS = [
    RidgeClassifierCVDecisionProba,
    RidgeClassifierCVIndicator,
    AutoSelectKBestClassifier,
    RareClassSafeLogisticCV,
]
SCALERS = [SparseScaler, NoScaler]


def _blas_threads():
    """Thread counts of the BLAS libraries threadpoolctl can see (e.g. OpenBLAS)."""
    return {lib["num_threads"] for lib in threadpool_info() if lib["user_api"] == "blas"}


needs_blas = pytest.mark.skipif(not _blas_threads(), reason="no BLAS visible to threadpoolctl")


def _make_data(n_classes=2, n_per_class=25, n_features=12, seed=0, str_labels=False):
    """Separable-but-noisy tabular blobs, one centroid per class."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_classes, n_features)) * 2.0
    X = np.repeat(centers, n_per_class, axis=0) + rng.standard_normal(
        (n_classes * n_per_class, n_features)
    )
    y = np.repeat(np.arange(n_classes), n_per_class)
    if str_labels:
        y = np.array([str(v) for v in y])
    return X, y


def _make_regression_data(n_samples=60, n_features=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))
    y = X @ rng.standard_normal(n_features) + 0.1 * rng.standard_normal(n_samples)
    return X, y


def _make_counts_data(n_samples=40, n_features=10, seed=0):
    """Sparse non-negative counts, the shape of feature block SparseScaler is built for."""
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 6, size=(n_samples, n_features)).astype(np.float32)
    X[rng.random(X.shape) < 0.5] = 0.0
    return X


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("str_labels", [False, True])
def test_predict_proba_rows_are_bounded_and_sum_to_one(head, n_classes, str_labels):
    X, y = _make_data(n_classes=n_classes, str_labels=str_labels)
    clf = head().fit(X, y)

    proba = clf.predict_proba(X)

    assert proba.shape == (X.shape[0], len(clf.classes_))
    assert np.isfinite(proba).all()
    assert (proba >= 0).all() and (proba <= 1).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("n_classes", [2, 3, 5])
@pytest.mark.parametrize("str_labels", [False, True])
def test_predict_proba_columns_follow_classes_order(head, n_classes, str_labels):
    X, y = _make_data(n_classes=n_classes, str_labels=str_labels)
    clf = head().fit(X, y)

    proba = clf.predict_proba(X)

    np.testing.assert_array_equal(clf.classes_[proba.argmax(axis=1)], clf.predict(X))


@pytest.mark.parametrize("head", HEADS)
def test_clone_preserves_constructor_params(head):
    clf = head(n_jobs=4)

    cloned = clone(clf)

    assert cloned.n_jobs == 4
    assert cloned.get_params() == clf.get_params()
    assert head().n_jobs == 1


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("n_jobs", [2, -1])
def test_n_jobs_does_not_change_the_fitted_model(head, n_jobs):
    X, y = _make_data(n_classes=3)
    baseline = head().fit(X, y)
    other = head(n_jobs=n_jobs).fit(X, y)

    # allclose, not exact: a different BLAS thread count reorders the reductions.
    np.testing.assert_allclose(
        other.predict_proba(X), baseline.predict_proba(X), rtol=1e-8, atol=1e-10
    )


@needs_blas
def test_logisticcv_fits_under_the_blas_limit(monkeypatch):
    seen = []
    original_fit = LogisticRegressionCV.fit

    def spy(self, X, y, **kwargs):
        seen.append(_blas_threads())
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(LogisticRegressionCV, "fit", spy)
    X, y = _make_data(n_classes=3)
    RareClassSafeLogisticCV(Cs=3, n_jobs=1).fit(X, y)

    assert seen == [{1}]


@needs_blas
def test_autoselect_feature_selection_runs_under_the_blas_limit():
    seen = []

    def score_func(X, y):
        seen.append(_blas_threads())
        return f_classif(X, y)

    X, y = _make_data(n_classes=3)
    AutoSelectKBestClassifier(k=5, score_func=score_func, n_jobs=1).fit(X, y)

    assert seen == [{1}]


@needs_blas
def test_mlp_fits_and_predicts_under_the_blas_limit(monkeypatch):
    seen = []
    original_fit = MLPClassifier.fit
    original_proba = MLPClassifier.predict_proba

    def spy_fit(self, X, y):
        seen.append(_blas_threads())
        return original_fit(self, X, y)

    def spy_proba(self, X):
        seen.append(_blas_threads())
        return original_proba(self, X)

    monkeypatch.setattr(MLPClassifier, "fit", spy_fit)
    monkeypatch.setattr(MLPClassifier, "predict_proba", spy_proba)
    X, y = _make_data(n_classes=3)
    clf = MLPClassifierThreadLimited(max_iter=50, random_state=0, n_jobs=1).fit(X, y)
    clf.predict_proba(X)

    assert seen == [{1}, {1}]


def test_mlp_thread_limited_matches_mlp():
    X, y = _make_data(n_classes=3)

    plain = MLPClassifier(max_iter=500, random_state=0).fit(X, y)
    limited = MLPClassifierThreadLimited(max_iter=500, random_state=0).fit(X, y)

    np.testing.assert_array_equal(limited.classes_, plain.classes_)
    np.testing.assert_allclose(
        limited.predict_proba(X), plain.predict_proba(X), rtol=1e-8, atol=1e-10
    )


def test_clone_preserves_mlp_thread_limited_params():
    clf = MLPClassifierThreadLimited(hidden_layer_sizes=(7,), max_iter=7, random_state=3, n_jobs=4)

    cloned = clone(clf)

    assert cloned.n_jobs == 4
    assert cloned.get_params() == clf.get_params()
    assert MLPClassifierThreadLimited().n_jobs == 1


def test_clipped_regressor_fits_and_predicts():
    X, y = _make_regression_data()

    reg = ClippedRegressor(regressor=RidgeCV()).fit(X, y)
    pred = reg.predict(X)

    assert pred.shape == (len(X),)
    assert np.isfinite(pred).all()


def test_clone_preserves_clipped_regressor_params():
    reg = ClippedRegressor(regressor=RidgeCV(alphas=(0.5, 5.0)))

    cloned = clone(reg)

    assert isinstance(cloned.regressor, RidgeCV)
    assert cloned.regressor.alphas == (0.5, 5.0)
    assert cloned.regressor is not reg.regressor


@pytest.mark.parametrize("scaler", SCALERS)
def test_scaler_transform_keeps_shape_and_stays_finite(scaler):
    X = _make_counts_data()

    Xt = scaler().fit(X).transform(X)

    assert Xt.shape == X.shape
    assert np.isfinite(Xt).all()


@pytest.mark.parametrize("scaler", SCALERS)
def test_scaler_does_not_mutate_its_input(scaler):
    X = _make_counts_data()
    original = X.copy()

    scaler().fit_transform(X)

    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize("scaler", SCALERS)
def test_scaler_fit_transform_matches_fit_then_transform(scaler):
    X = _make_counts_data()

    one_pass = scaler().fit_transform(X)
    two_pass = scaler().fit(X).transform(X)

    np.testing.assert_allclose(one_pass, two_pass)

