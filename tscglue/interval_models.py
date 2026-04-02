"""RSTSF variants: RandomIntervals and UnsupervisedIntervals + GaussianRandomProjection."""

__all__ = ["UnsupervisedIntervals", "RSTSFRandom", "RSTSFRandomTransformer", "RSTSFUnsupervised", "RSTSFUnsupervisedRaw", "RSTSFCombined", "RSTSFCombinedTransformer", "RSTSFUnsupervisedPerRep"]

import inspect

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
from aeon.classification import BaseClassifier
from aeon.transformations.base import BaseTransformer
from aeon.transformations.collection import (
    ARCoefficientTransformer,
    PeriodogramTransformer,
)
from aeon.transformations.collection.base import BaseCollectionTransformer
from aeon.transformations.collection.interval_based import RandomIntervals
from aeon.utils.numba.general import first_order_differences_3d, z_normalise_series_3d
from aeon.utils.numba.stats import (
    row_count_above_mean,
    row_count_mean_crossing,
    row_iqr,
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_slope,
    row_std,
)
from aeon.utils.validation import check_n_jobs
from tscglue.models_tsfm import RidgeClassifierCVDecisionProba


class UnsupervisedIntervals(BaseCollectionTransformer):
    """Unsupervised interval feature transformer.

    Extracts interval features using the same recursive splitting idea as
    SupervisedIntervals, but instead of keeping only the better-scoring half,
    it keeps both halves and continues recursively on both branches.

    Parameters
    ----------
    n_intervals : int, default=50
        Number of interval-generation runs per channel/feature pair.
    min_interval_length : int, default=3
        Minimum length of extracted intervals.
    features : callable or list of callables, default=None
        Feature functions. If None, uses [mean, median, std, slope, min, max,
        iqr, count_mean_crossing, count_above_mean].
    randomised_split_point : bool, default=True
        If True, recursive splits are randomised subject to min_interval_length.
    normalise_for_search : bool, default=True
        If True, splitting uses z-normalised data but features are computed on
        the original data.
    random_state : None, int or RandomState, default=None
    n_jobs : int, default=1
    parallel_backend : str or None, default=None
    """

    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "requires_y": False,
        "algorithm_type": "interval",
    }

    transformer_feature_selection = ["features"]
    transformer_feature_names = [
        "features_arguments_",
        "_features_arguments",
        "get_features_arguments",
        "_get_features_arguments",
    ]

    def __init__(
        self,
        n_intervals=50,
        min_interval_length=3,
        features=None,
        randomised_split_point=True,
        normalise_for_search=True,
        random_state=None,
        n_jobs=1,
        parallel_backend=None,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.features = features
        self.randomised_split_point = randomised_split_point
        self.normalise_for_search = normalise_for_search
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend
        super().__init__()

    def _fit(self, X, y=None):
        X, rng = self._fit_setup(X)
        X_norm = z_normalise_series_3d(X) if self.normalise_for_search else X

        fit = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._generate_intervals)(
                X, X_norm, rng.randint(np.iinfo(np.int32).max)
            )
            for _ in range(self.n_intervals)
        )

        for ints in fit:
            self.intervals_.extend(ints)
        self._transform_features = [True] * len(self.intervals_)
        return self

    def _transform(self, X, y=None):
        transform = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._transform_intervals)(X, i)
            for i in range(len(self.intervals_))
        )

        Xt = np.zeros((X.shape[0], len(transform)))
        for i, t in enumerate(transform):
            Xt[:, i] = t
        return Xt

    def _fit_setup(self, X):
        self.intervals_ = []
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        if self.n_cases_ <= 0:
            raise ValueError("UnsupervisedIntervals requires at least 1 time series.")

        self._min_interval_length = max(3, self.min_interval_length)
        if self._min_interval_length * 2 > self.n_timepoints_:
            raise ValueError(
                "Minimum interval length must allow at least one valid split."
            )

        self._features = self.features
        if self.features is None:
            self._features = [
                row_mean,
                row_median,
                row_std,
                row_slope,
                row_numba_min,
                row_numba_max,
                row_iqr,
                row_count_mean_crossing,
                row_count_above_mean,
            ]
        if not isinstance(self._features, list):
            self._features = [self._features]

        rng = check_random_state(self.random_state)

        msg = (
            "Transformers must have a parameter from 'transformer_feature_selection' "
            "and a list of feature names in 'transformer_feature_names'. "
            "Transformers which require 'fit' are currently unsupported."
        )

        expanded_features = []
        for f in self._features:
            if callable(f):
                expanded_features.append(f)
            elif isinstance(f, BaseTransformer):
                if not f.get_tag("fit_is_empty"):
                    raise ValueError(msg)
                params = inspect.signature(f.__init__).parameters
                att_name = None
                for n in self.transformer_feature_selection:
                    if params.get(n, None) is not None:
                        att_name = n
                        break
                if att_name is None:
                    raise ValueError(msg)
                t_features = None
                for n in self.transformer_feature_names:
                    if hasattr(f, n) and isinstance(getattr(f, n), (list, tuple)):
                        t_features = getattr(f, n)
                        break
                if t_features is None:
                    raise ValueError(msg)
                for t_f in t_features:
                    new_transformer = _clone_estimator(f, rng)
                    setattr(new_transformer, att_name, t_f)
                    expanded_features.append(new_transformer)
            else:
                raise ValueError("Features must be callables or BaseTransformer instances.")

        self._features = expanded_features
        return X, rng

    def _generate_intervals(self, X, X_norm, seed):
        rng = check_random_state(seed)
        intervals = []

        for dim in range(self.n_channels_):
            for feature in self._features:
                random_cut_point = int(rng.randint(1, self.n_timepoints_ - 1))
                while (
                    random_cut_point < self._min_interval_length
                    or self.n_timepoints_ - random_cut_point < self._min_interval_length
                ):
                    random_cut_point = int(rng.randint(1, self.n_timepoints_ - 1))

                intervals.extend(self._unsupervised_search(
                    X_norm[:, dim, :random_cut_point], 0, feature, dim, rng,
                ))
                intervals.extend(self._unsupervised_search(
                    X_norm[:, dim, random_cut_point:], random_cut_point, feature, dim, rng,
                ))

        return intervals

    def _transform_intervals(self, X, idx):
        if not self._transform_features[idx]:
            return np.zeros(X.shape[0])
        start, end, dim, feature = self.intervals_[idx]
        if isinstance(feature, BaseTransformer):
            return feature.transform(X[:, dim, start:end]).flatten()
        return feature(X[:, dim, start:end])

    def _unsupervised_search(self, X, ini_idx, feature, dim, rng):
        if X.shape[1] < self._min_interval_length * 2:
            return []

        if self.randomised_split_point and X.shape[1] != self._min_interval_length * 2:
            div_point = rng.randint(
                self._min_interval_length, X.shape[1] - self._min_interval_length
            )
        else:
            div_point = int(X.shape[1] / 2)

        start_0, end_0 = ini_idx, ini_idx + div_point
        start_1, end_1 = ini_idx + div_point, ini_idx + X.shape[1]

        intervals = [(start_0, end_0, dim, feature), (start_1, end_1, dim, feature)]
        intervals.extend(self._unsupervised_search(X[:, :div_point], start_0, feature, dim, rng))
        intervals.extend(self._unsupervised_search(X[:, div_point:], start_1, feature, dim, rng))
        return intervals

    def set_features_to_transform(self, arr, raise_error=True):
        if len(arr) != len(self.intervals_) or not all(isinstance(b, bool) for b in arr):
            if raise_error:
                raise ValueError(
                    "Input must be a list of bools of length len(intervals_)."
                )
            return False
        self._transform_features = arr
        return True

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        if parameter_set == "results_comparison":
            return {"n_intervals": 1, "randomised_split_point": True}
        return {"n_intervals": 1, "randomised_split_point": False}


def _make_series_transforms(X):
    """Return (original, diff, periodogram, AR) representations."""
    lags = int(12 * (X.shape[2] / 100.0) ** 0.25)
    series_transformers = [
        FunctionTransformer(func=first_order_differences_3d, validate=False),
        PeriodogramTransformer(),
        ARCoefficientTransformer(order=lags, replace_nan=True),
    ]
    transforms = [X] + [t.fit_transform(X) for t in series_transformers]
    return transforms, series_transformers


def _build_et(n_estimators, n_jobs, random_state):
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        criterion="entropy",
        class_weight="balanced",
        max_features="sqrt",
        n_jobs=n_jobs,
        random_state=random_state,
    )


def _build_ridge():
    return RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10))


class RSTSFRandom(BaseClassifier):
    """RSTSF using RandomIntervals instead of SupervisedIntervals.

    Same 4 series representations (original, first differences, periodogram,
    AR coefficients) as RSTSF, but interval extraction is random rather than
    supervised.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees when estimator is None (ExtraTreesClassifier).
        Ignored if estimator is provided.
    n_intervals : int, default=600
        Number of random intervals per series representation.
    min_interval_length : int, default=3
        Minimum length of extracted intervals.
    estimator : sklearn estimator or None, default=None
        Classifier fitted on the interval features. None defaults to
        ExtraTreesClassifier (entropy, balanced, sqrt). Feature pruning via
        tree importance is only applied when estimator is None.
    random_state : None, int or RandomState, default=None
    n_jobs : int, default=1
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        n_estimators=200,
        n_intervals=600,
        min_interval_length=3,
        estimator=None,
        random_state=None,
        n_jobs=1,
    ):
        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs
        super().__init__()

    def _fit(self, X, y):
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        transforms, self._series_transformers = _make_series_transforms(X)

        Xt = np.empty((X.shape[0], 0))
        self._transformers = []
        transform_data_lengths = []
        for t in transforms:
            ri = RandomIntervals(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
            )
            features = ri.fit_transform(t)
            Xt = np.hstack((Xt, features))
            self._transformers.append(ri)
            transform_data_lengths.append(features.shape[1])

        use_et = self.estimator is None
        self.clf_ = (
            _build_et(self.n_estimators, self._n_jobs, self.random_state)
            if use_et
            else clone(self.estimator)
        )
        self.clf_.fit(Xt, y)

        if use_et:
            relevant_features = np.unique(
                [f for tree in self.clf_.estimators_ for f in tree.tree_.feature[tree.tree_.feature >= 0]]
            )
            features_to_transform = [False] * Xt.shape[1]
            for i in relevant_features:
                features_to_transform[i] = True
            count = 0
            for r in range(len(transforms)):
                self._transformers[r].set_features_to_transform(
                    features_to_transform[count : count + transform_data_lengths[r]],
                    raise_error=False,
                )
                count += transform_data_lengths[r]

        return self

    def _predict(self, X):
        return self.clf_.predict(self._predict_transform(X))

    def _predict_proba(self, X):
        return self.clf_.predict_proba(self._predict_transform(X))

    def _predict_transform(self, X):
        transforms = [X] + [t.transform(X) for t in self._series_transformers]
        Xt = np.empty((X.shape[0], 0))
        for i, t in enumerate(transforms):
            Xt = np.hstack((Xt, self._transformers[i].transform(t)))
        return Xt

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"n_estimators": 2, "n_intervals": 5}


class RSTSFRandomTransformer(BaseEstimator, TransformerMixin):
    """Feature extraction stage of RSTSFRandom, usable as a standalone transformer.

    Applies RandomIntervals to the same 4 series representations as RSTSF
    (original, first differences, periodogram, AR coefficients). The result is
    a fixed-width feature matrix that can be fed to any downstream classifier.

    Parameters
    ----------
    n_intervals : int, default=600
    min_interval_length : int, default=3
    random_state : None, int or RandomState, default=None
    n_jobs : int, default=1
    """

    def __init__(self, n_intervals=600, min_interval_length=3, random_state=None, n_jobs=1):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self._n_jobs = check_n_jobs(self.n_jobs)
        transforms, self._series_transformers = _make_series_transforms(X)
        self._ri_transformers = []
        for t in transforms:
            ri = RandomIntervals(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
            )
            ri.fit(t)
            self._ri_transformers.append(ri)
        return self

    def transform(self, X):
        transforms = [X] + [t.transform(X) for t in self._series_transformers]
        return np.hstack([ri.transform(t) for ri, t in zip(self._ri_transformers, transforms)])


class RSTSFUnsupervised(BaseClassifier):
    """RSTSF-style classifier using UnsupervisedIntervals + GaussianRandomProjection.

    Applies UnsupervisedIntervals to the same 4 series representations as RSTSF
    (original, first differences, periodogram, AR coefficients), concatenates all
    features, reduces dimensionality with GaussianRandomProjection, then trains
    a classifier.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees when estimator is None (ExtraTreesClassifier).
        Ignored if estimator is provided.
    n_intervals : int, default=50
        Number of interval-generation runs per series representation.
    min_interval_length : int, default=3
        Minimum length of extracted intervals.
    n_components : int or "auto", default="auto"
        Number of components for GaussianRandomProjection.
        "auto" uses the Johnson-Lindenstrauss lemma.
    estimator : sklearn estimator or None, default=None
        Classifier fitted on the projected features. None defaults to
        ExtraTreesClassifier (entropy, balanced, sqrt).
    random_state : None, int or RandomState, default=None
    n_jobs : int, default=1
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        n_estimators=200,
        n_intervals=50,
        min_interval_length=3,
        n_components="auto",
        estimator=None,
        random_state=None,
        n_jobs=1,
        verbose=False,
    ):
        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.n_components = n_components
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        super().__init__()

    def _fit(self, X, y):
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        transforms, self._series_transformers = _make_series_transforms(X)

        Xt = np.empty((X.shape[0], 0))
        self._transformers = []
        for i, t in enumerate(transforms):
            ui = UnsupervisedIntervals(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
            )
            features = ui.fit_transform(t)
            if self.verbose:
                print(f"[RSTSFUnsupervised] rep {i} features: {features.shape} ({features.nbytes / 1024**2:.1f} MB)")
            Xt = np.hstack((Xt, features))
            self._transformers.append(ui)

        if self.verbose:
            print(f"[RSTSFUnsupervised] Xt before GRP: {Xt.shape} ({Xt.nbytes / 1024**2:.1f} MB)")
        n_components = self.n_components
        if n_components == "auto":
            from sklearn.random_projection import johnson_lindenstrauss_min_dim
            n_components = min(
                johnson_lindenstrauss_min_dim(Xt.shape[0], eps=0.1),
                Xt.shape[1],
            )
        self.rp_ = GaussianRandomProjection(
            n_components=n_components,
            random_state=self.random_state,
        )
        Xt = self.rp_.fit_transform(Xt)
        if self.verbose:
            print(f"[RSTSFUnsupervised] Xt after GRP: {Xt.shape} ({Xt.nbytes / 1024**2:.1f} MB)")

        self.clf_ = (
            _build_et(self.n_estimators, self._n_jobs, self.random_state)
            if self.estimator is None
            else clone(self.estimator)
        )
        self.clf_.fit(Xt, y)
        return self

    def _predict(self, X):
        return self.clf_.predict(self._predict_transform(X))

    def _predict_proba(self, X):
        return self.clf_.predict_proba(self._predict_transform(X))

    def _predict_transform(self, X):
        transforms = [X] + [t.transform(X) for t in self._series_transformers]
        Xt = np.empty((X.shape[0], 0))
        for i, t in enumerate(transforms):
            Xt = np.hstack((Xt, self._transformers[i].transform(t)))
        return self.rp_.transform(Xt)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"n_estimators": 2, "n_intervals": 1}


class RSTSFUnsupervisedRaw(BaseClassifier):
    """RSTSF-style classifier using UnsupervisedIntervals with RidgeCV, no GRP.

    Applies UnsupervisedIntervals to the same 4 series representations as RSTSF
    (original, first differences, periodogram, AR coefficients), concatenates all
    features, and trains a RidgeClassifierCV directly on the raw feature matrix
    without any dimensionality reduction.

    Parameters
    ----------
    n_intervals : int, default=50
        Number of interval-generation runs per series representation.
    min_interval_length : int, default=3
        Minimum length of extracted intervals.
    random_state : None, int or RandomState, default=None
    n_jobs : int, default=1
    verbose : bool, default=False
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        n_intervals=50,
        min_interval_length=3,
        random_state=None,
        n_jobs=1,
        verbose=False,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        super().__init__()

    def _fit(self, X, y):
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        transforms, self._series_transformers = _make_series_transforms(X)

        Xt = np.empty((X.shape[0], 0))
        self._transformers = []
        for i, t in enumerate(transforms):
            ui = UnsupervisedIntervals(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
            )
            features = ui.fit_transform(t)
            if self.verbose:
                print(f"[RSTSFUnsupervisedRaw] rep {i} features: {features.shape} ({features.nbytes / 1024**2:.1f} MB)")
            Xt = np.hstack((Xt, features))
            self._transformers.append(ui)

        if self.verbose:
            print(f"[RSTSFUnsupervisedRaw] Xt: {Xt.shape} ({Xt.nbytes / 1024**2:.1f} MB)")
        self.scaler_ = StandardScaler()
        Xt = self.scaler_.fit_transform(Xt)
        self.clf_ = _build_ridge()
        self.clf_.fit(Xt, y)
        return self

    def _predict(self, X):
        return self.clf_.predict(self._predict_transform(X))

    def _predict_proba(self, X):
        return self.clf_.predict_proba(self._predict_transform(X))

    def _predict_transform(self, X):
        transforms = [X] + [t.transform(X) for t in self._series_transformers]
        Xt = np.empty((X.shape[0], 0))
        for i, t in enumerate(transforms):
            Xt = np.hstack((Xt, self._transformers[i].transform(t)))
        return self.scaler_.transform(Xt)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"n_intervals": 1}


class RSTSFCombinedTransformer(BaseEstimator, TransformerMixin):
    """Feature extraction stage of RSTSFCombined, usable as a standalone transformer.

    Applies RandomIntervals and UnsupervisedIntervals to the same 4 series
    representations as RSTSF (original, first differences, periodogram, AR
    coefficients). Unsupervised features are compressed with
    GaussianRandomProjection, then concatenated with the random interval features.
    The result is a fixed-width feature matrix that can be fed to any downstream
    classifier (e.g. RidgeCV with StandardScaler).

    Parameters
    ----------
    n_intervals_random : int, default=600
    n_intervals_unsupervised : int, default=50
    min_interval_length : int, default=3
    n_components : int or "auto", default="auto"
    random_state : None, int or RandomState, default=None
    n_jobs : int, default=1
    verbose : bool, default=False
    """

    def __init__(
        self,
        n_intervals_random=600,
        n_intervals_unsupervised=50,
        min_interval_length=3,
        n_components="auto",
        random_state=None,
        n_jobs=1,
        verbose=False,
    ):
        self.n_intervals_random = n_intervals_random
        self.n_intervals_unsupervised = n_intervals_unsupervised
        self.min_interval_length = min_interval_length
        self.n_components = n_components
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[RSTSFCombinedTransformer] {msg}")

    def fit(self, X, y=None):
        from time import perf_counter
        self._n_jobs = check_n_jobs(self.n_jobs)
        t0 = perf_counter()
        transforms, self._series_transformers = _make_series_transforms(X)
        self._log(f"series transforms: {perf_counter() - t0:.2f}s")

        self._ri_transformers = []
        self._ui_transformers = []

        rep_names = ["raw", "diff", "periodogram", "ar"]
        Xt_unsup_probe = []
        for i, t in enumerate(transforms):
            t0 = perf_counter()
            ri = RandomIntervals(
                n_intervals=self.n_intervals_random,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
            )
            ri.fit(t)
            self._ri_transformers.append(ri)
            self._log(f"fit RI  rep={rep_names[i]}: {perf_counter() - t0:.2f}s")

            t0 = perf_counter()
            ui = UnsupervisedIntervals(
                n_intervals=self.n_intervals_unsupervised,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
            )
            ui.fit(t[:1])
            self._ui_transformers.append(ui)
            Xt_unsup_probe.append(ui.transform(t[:1]))
            self._log(f"fit UI  rep={rep_names[i]}: {perf_counter() - t0:.2f}s")

        t0 = perf_counter()
        probe_unsup = np.hstack(Xt_unsup_probe)
        n_components = self.n_components
        if n_components == "auto":
            from sklearn.random_projection import johnson_lindenstrauss_min_dim
            n_components = min(
                johnson_lindenstrauss_min_dim(X.shape[0], eps=0.1),
                probe_unsup.shape[1],
            )
        self.rp_ = GaussianRandomProjection(n_components=n_components, random_state=self.random_state)
        self.rp_.fit(probe_unsup)  # GRP only needs n_features from shape, not actual values
        self._log(f"fit GRP (unsup_features={probe_unsup.shape[1]} -> n_components={n_components}): {perf_counter() - t0:.2f}s")
        return self

    def transform(self, X):
        from time import perf_counter
        rep_names = ["raw", "diff", "periodogram", "ar"]
        t0 = perf_counter()
        transforms = [X] + [t.transform(X) for t in self._series_transformers]
        self._log(f"transform series reps: {perf_counter() - t0:.2f}s")

        rand_parts, unsup_parts = [], []
        for i, t in enumerate(transforms):
            t0 = perf_counter()
            rand_parts.append(self._ri_transformers[i].transform(t))
            self._log(f"transform RI  rep={rep_names[i]}: {perf_counter() - t0:.2f}s")

            t0 = perf_counter()
            unsup_parts.append(self._ui_transformers[i].transform(t))
            self._log(f"transform UI  rep={rep_names[i]}: {perf_counter() - t0:.2f}s")

        t0 = perf_counter()
        Xt_rand = np.hstack(rand_parts)
        Xt_unsup_proj = self.rp_.transform(np.hstack(unsup_parts))
        self._log(f"GRP projection: {perf_counter() - t0:.2f}s  output={np.hstack((Xt_rand, Xt_unsup_proj)).shape}")
        return np.hstack((Xt_rand, Xt_unsup_proj))


class RSTSFCombined(BaseClassifier):
    """RSTSF-style classifier combining RandomIntervals and UnsupervisedIntervals.

    For each of the 4 series representations (original, first differences,
    periodogram, AR coefficients), extracts features from both RandomIntervals
    and UnsupervisedIntervals. The unsupervised features are first projected
    with GaussianRandomProjection, then concatenated with the random interval
    features before fitting the classifier.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees when estimator is None (ExtraTreesClassifier).
        Ignored if estimator is provided.
    n_intervals_random : int, default=600
        Number of random intervals per series representation.
    n_intervals_unsupervised : int, default=50
        Number of unsupervised interval-generation runs per series representation.
    min_interval_length : int, default=3
        Minimum length of extracted intervals.
    n_components : int or "auto", default="auto"
        Number of GaussianRandomProjection components applied to the
        unsupervised features before concatenation.
    estimator : sklearn estimator or None, default=None
        Classifier fitted on the combined features. None defaults to
        ExtraTreesClassifier (entropy, balanced, sqrt). Feature pruning via
        tree importance is only applied when estimator is None.
    random_state : None, int or RandomState, default=None
    n_jobs : int, default=1
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        n_estimators=200,
        n_intervals_random=600,
        n_intervals_unsupervised=50,
        min_interval_length=3,
        n_components="auto",
        estimator=None,
        random_state=None,
        n_jobs=1,
        verbose=False,
    ):
        self.n_estimators = n_estimators
        self.n_intervals_random = n_intervals_random
        self.n_intervals_unsupervised = n_intervals_unsupervised
        self.min_interval_length = min_interval_length
        self.n_components = n_components
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        super().__init__()

    def _fit(self, X, y):
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        transforms, self._series_transformers = _make_series_transforms(X)

        self._ri_transformers = []
        self._ui_transformers = []
        Xt_rand = np.empty((X.shape[0], 0))
        Xt_unsup = np.empty((X.shape[0], 0))
        ri_lengths = []

        for i, t in enumerate(transforms):
            ri = RandomIntervals(
                n_intervals=self.n_intervals_random,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
            )
            ri_features = ri.fit_transform(t)
            if self.verbose:
                print(f"[RSTSFCombined] rep {i} rand features: {ri_features.shape} ({ri_features.nbytes / 1024**2:.1f} MB)")
            Xt_rand = np.hstack((Xt_rand, ri_features))
            self._ri_transformers.append(ri)
            ri_lengths.append(ri_features.shape[1])

            ui = UnsupervisedIntervals(
                n_intervals=self.n_intervals_unsupervised,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
            )
            ui_features = ui.fit_transform(t)
            if self.verbose:
                print(f"[RSTSFCombined] rep {i} unsup features: {ui_features.shape} ({ui_features.nbytes / 1024**2:.1f} MB)")
            Xt_unsup = np.hstack((Xt_unsup, ui_features))
            self._ui_transformers.append(ui)

        if self.verbose:
            print(f"[RSTSFCombined] Xt_rand before concat: {Xt_rand.shape} ({Xt_rand.nbytes / 1024**2:.1f} MB)")
            print(f"[RSTSFCombined] Xt_unsup before GRP: {Xt_unsup.shape} ({Xt_unsup.nbytes / 1024**2:.1f} MB)")
        n_components = self.n_components
        if n_components == "auto":
            from sklearn.random_projection import johnson_lindenstrauss_min_dim
            n_components = min(
                johnson_lindenstrauss_min_dim(Xt_unsup.shape[0], eps=0.1),
                Xt_unsup.shape[1],
            )
        self.rp_ = GaussianRandomProjection(
            n_components=n_components,
            random_state=self.random_state,
        )
        Xt_unsup_proj = self.rp_.fit_transform(Xt_unsup)
        if self.verbose:
            print(f"[RSTSFCombined] Xt_unsup after GRP: {Xt_unsup_proj.shape} ({Xt_unsup_proj.nbytes / 1024**2:.1f} MB)")

        Xt = np.hstack((Xt_rand, Xt_unsup_proj))

        use_et = self.estimator is None
        self.clf_ = (
            _build_et(self.n_estimators, self._n_jobs, self.random_state)
            if use_et
            else clone(self.estimator)
        )
        self.clf_.fit(Xt, y)

        if use_et:
            n_rand_total = Xt_rand.shape[1]
            relevant_features = np.unique(
                [f for tree in self.clf_.estimators_ for f in tree.tree_.feature[tree.tree_.feature >= 0]]
            )
            # only prune the random interval features (unsupervised are projected)
            rand_relevant = relevant_features[relevant_features < n_rand_total]
            features_to_transform = [False] * n_rand_total
            for i in rand_relevant:
                features_to_transform[i] = True
            count = 0
            for r in range(len(transforms)):
                self._ri_transformers[r].set_features_to_transform(
                    features_to_transform[count : count + ri_lengths[r]],
                    raise_error=False,
                )
                count += ri_lengths[r]

        return self

    def _predict(self, X):
        return self.clf_.predict(self._predict_transform(X))

    def _predict_proba(self, X):
        return self.clf_.predict_proba(self._predict_transform(X))

    def _predict_transform(self, X):
        transforms = [X] + [t.transform(X) for t in self._series_transformers]
        Xt_rand = np.empty((X.shape[0], 0))
        Xt_unsup = np.empty((X.shape[0], 0))
        for i, t in enumerate(transforms):
            Xt_rand = np.hstack((Xt_rand, self._ri_transformers[i].transform(t)))
            Xt_unsup = np.hstack((Xt_unsup, self._ui_transformers[i].transform(t)))
        return np.hstack((Xt_rand, self.rp_.transform(Xt_unsup)))

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"n_estimators": 2, "n_intervals_random": 5, "n_intervals_unsupervised": 1}


def _resolve_n_components(n_components, n_samples, n_features):
    """Resolve n_components for GRP, capping at n_features when "auto"."""
    if n_components != "auto":
        return n_components
    from sklearn.random_projection import johnson_lindenstrauss_min_dim
    return int(min(johnson_lindenstrauss_min_dim(n_samples, eps=0.1), n_features))


class RSTSFUnsupervisedPerRep(BaseClassifier):
    """RSTSFUnsupervised with a single GRP and sample-batched projection.

    Identical algorithm to RSTSFUnsupervised (4 series representations →
    UnsupervisedIntervals → single GaussianRandomProjection → ExtraTreesClassifier)
    but avoids materialising the full (n_samples × total_features) matrix by
    processing ``batch_size`` samples at a time:

      for each batch of rows:
          compute all 4 series representations
          concatenate interval features
          transform through GRP
          store projected batch

    The GRP is first fitted on a single-sample probe (only the feature-count
    matters for generating the random projection matrix), with ``n_components``
    capped at ``n_features`` when ``"auto"`` to avoid the JL-lemma overshoot
    error on small datasets.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees when estimator is None (ExtraTreesClassifier).
    n_intervals : int, default=50
        UnsupervisedIntervals runs per series representation.
    min_interval_length : int, default=3
        Minimum length of extracted intervals.
    n_components : int or "auto", default="auto"
        GRP output dimension.  ``"auto"`` uses the Johnson-Lindenstrauss lemma
        capped at the input feature count.
    batch_size : int, default=50
        Number of samples processed per projection batch.
    estimator : sklearn estimator or None, default=None
        Classifier fitted on the projected features.
    random_state : None, int or RandomState, default=None
    n_jobs : int, default=1
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "interval",
        "python_dependencies": "statsmodels",
    }

    def __init__(
        self,
        n_estimators=200,
        n_intervals=50,
        min_interval_length=3,
        n_components="auto",
        batch_size=50,
        estimator=None,
        random_state=None,
        n_jobs=1,
        verbose=False,
    ):
        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.n_components = n_components
        self.batch_size = batch_size
        self.estimator = estimator
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        super().__init__()

    def _fit(self, X, y):
        self.n_cases_, self.n_channels_, self.n_timepoints_ = X.shape
        self._n_jobs = check_n_jobs(self.n_jobs)

        # Build series transformers without transforming the full dataset all at once.
        lags = int(12 * (X.shape[2] / 100.0) ** 0.25)
        self._series_transformers = [
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(),
            ARCoefficientTransformer(order=lags, replace_nan=True),
        ]

        # Fit each UI one representation at a time so only one full-dataset
        # transform is in memory at once.
        self._transformers = []
        for i, rep in enumerate([None] + self._series_transformers):
            t = rep.fit_transform(X) if rep is not None else X
            ui = UnsupervisedIntervals(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
            )
            ui.fit(t[:1])
            self._transformers.append(ui)
            if rep is not None:
                del t  # free transform before computing the next rep

        # Probe one sample to determine feature count, then fit GRP.
        probe = self._transform_batch(X[:1])
        n_features = probe.shape[1]
        n_comp = _resolve_n_components(self.n_components, self.n_cases_, n_features)
        self.rp_ = GaussianRandomProjection(n_components=n_comp, random_state=self.random_state)
        self.rp_.fit(probe)  # only shape matters; projection matrix is random
        n_batches = (self.n_cases_ + self.batch_size - 1) // self.batch_size
        if self.verbose:
            print(f"[RSTSFUnsupervisedPerRep] n_features={n_features} -> n_components={n_comp}, n_batches={n_batches}")

        # For each batch: compute series representations, extract features, project.
        projected_batches = []
        for start in range(0, self.n_cases_, self.batch_size):
            sl = slice(start, start + self.batch_size)
            batch_features = self._transform_batch(X[sl])
            batch_idx = start // self.batch_size
            projected = self.rp_.transform(batch_features)
            if self.verbose:
                print(f"[RSTSFUnsupervisedPerRep] fit batch {batch_idx+1}/{n_batches}  samples={batch_features.shape[0]}  {batch_features.shape} ({batch_features.nbytes / 1024**2:.1f} MB) -> {projected.shape} ({projected.nbytes / 1024**2:.1f} MB)")
            projected_batches.append(projected)
        Xt = np.vstack(projected_batches)
        if self.verbose:
            print(f"[RSTSFUnsupervisedPerRep] fit done: Xt={Xt.shape} ({Xt.nbytes / 1024**2:.1f} MB)")

        self.clf_ = (
            _build_et(self.n_estimators, self._n_jobs, self.random_state)
            if self.estimator is None
            else clone(self.estimator)
        )
        self.clf_.fit(Xt, y)
        return self

    def _transform_batch(self, X_batch):
        """Compute all series representations and UI features for a batch of samples."""
        reps = [X_batch] + [t.transform(X_batch) for t in self._series_transformers]
        return np.hstack([self._transformers[i].transform(r) for i, r in enumerate(reps)])

    def _predict(self, X):
        return self.clf_.predict(self._predict_transform(X))

    def _predict_proba(self, X):
        return self.clf_.predict_proba(self._predict_transform(X))

    def _predict_transform(self, X):
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        projected_batches = []
        for start in range(0, n_samples, self.batch_size):
            sl = slice(start, start + self.batch_size)
            batch_features = self._transform_batch(X[sl])
            batch_idx = start // self.batch_size
            if self.verbose:
                print(f"[RSTSFUnsupervisedPerRep] predict batch {batch_idx+1}/{n_batches}  features={batch_features.shape} ({batch_features.nbytes / 1024**2:.1f} MB)")
            projected_batches.append(self.rp_.transform(batch_features))
        return np.vstack(projected_batches)

    @classmethod
    def _get_test_params(cls, parameter_set="default"):
        return {"n_estimators": 2, "n_intervals": 1}
