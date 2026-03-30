"""RSTSF variants: RandomIntervals and UnsupervisedIntervals + GaussianRandomProjection."""

__all__ = ["UnsupervisedIntervals", "RSTSFRandom", "RSTSFUnsupervised"]

import inspect

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import FunctionTransformer
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

    def _fit_transform(self, X, y=None):
        X, rng = self._fit_setup(X)
        X_norm = z_normalise_series_3d(X) if self.normalise_for_search else X

        fit = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._generate_intervals)(
                X, X_norm, rng.randint(np.iinfo(np.int32).max), True
            )
            for _ in range(self.n_intervals)
        )

        intervals, transformed_intervals = zip(*fit)
        for ints in intervals:
            self.intervals_.extend(ints)
        self._transform_features = [True] * len(self.intervals_)

        Xt = transformed_intervals[0]
        for i in range(1, self.n_intervals):
            Xt = np.hstack((Xt, transformed_intervals[i]))
        return Xt

    def _fit(self, X, y=None):
        X, rng = self._fit_setup(X)
        X_norm = z_normalise_series_3d(X) if self.normalise_for_search else X

        fit = Parallel(
            n_jobs=self._n_jobs, backend=self.parallel_backend, prefer="threads"
        )(
            delayed(self._generate_intervals)(
                X, X_norm, rng.randint(np.iinfo(np.int32).max), False
            )
            for _ in range(self.n_intervals)
        )

        intervals, _ = zip(*fit)
        for ints in intervals:
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

    def _generate_intervals(self, X, X_norm, seed, keep_transform):
        rng = check_random_state(seed)
        Xt = np.empty((self.n_cases_, 0)) if keep_transform else None
        intervals = []

        for dim in range(self.n_channels_):
            for feature in self._features:
                random_cut_point = int(rng.randint(1, self.n_timepoints_ - 1))
                while (
                    random_cut_point < self._min_interval_length
                    or self.n_timepoints_ - random_cut_point < self._min_interval_length
                ):
                    random_cut_point = int(rng.randint(1, self.n_timepoints_ - 1))

                intervals_L, Xt_L = self._unsupervised_search(
                    X_norm[:, dim, :random_cut_point],
                    0, feature, dim, X[:, dim, :], rng, keep_transform,
                    isinstance(feature, BaseTransformer),
                )
                intervals.extend(intervals_L)
                if keep_transform and Xt_L.shape[1] > 0:
                    Xt = np.hstack((Xt, Xt_L))

                intervals_R, Xt_R = self._unsupervised_search(
                    X_norm[:, dim, random_cut_point:],
                    random_cut_point, feature, dim, X[:, dim, :], rng, keep_transform,
                    isinstance(feature, BaseTransformer),
                )
                intervals.extend(intervals_R)
                if keep_transform and Xt_R.shape[1] > 0:
                    Xt = np.hstack((Xt, Xt_R))

        return intervals, Xt

    def _transform_intervals(self, X, idx):
        if not self._transform_features[idx]:
            return np.zeros(X.shape[0])
        start, end, dim, feature = self.intervals_[idx]
        if isinstance(feature, BaseTransformer):
            return feature.transform(X[:, dim, start:end]).flatten()
        return feature(X[:, dim, start:end])

    def _compute_feature(self, feature, X_interval, feature_is_transformer):
        if feature_is_transformer:
            return feature.transform(X_interval).flatten()
        return feature(X_interval)

    def _unsupervised_search(
        self, X, ini_idx, feature, dim, X_ori, rng, keep_transform, feature_is_transformer
    ):
        intervals = []
        Xt = np.empty((X.shape[0], 0)) if keep_transform else None

        if X.shape[1] < self._min_interval_length * 2:
            return intervals, Xt

        if self.randomised_split_point and X.shape[1] != self._min_interval_length * 2:
            div_point = rng.randint(
                self._min_interval_length, X.shape[1] - self._min_interval_length
            )
        else:
            div_point = int(X.shape[1] / 2)

        start_0, end_0 = ini_idx, ini_idx + div_point
        start_1, end_1 = ini_idx + div_point, ini_idx + X.shape[1]

        intervals.append((start_0, end_0, dim, feature))
        intervals.append((start_1, end_1, dim, feature))

        if keep_transform:
            feat_0 = self._compute_feature(feature, X_ori[:, start_0:end_0], feature_is_transformer)
            feat_1 = self._compute_feature(feature, X_ori[:, start_1:end_1], feature_is_transformer)
            Xt = np.hstack((Xt, feat_0.reshape(-1, 1), feat_1.reshape(-1, 1)))

        intervals_L, Xt_L = self._unsupervised_search(
            X[:, :div_point], start_0, feature, dim, X_ori, rng, keep_transform, feature_is_transformer
        )
        intervals.extend(intervals_L)
        if keep_transform and Xt_L.shape[1] > 0:
            Xt = np.hstack((Xt, Xt_L))

        intervals_R, Xt_R = self._unsupervised_search(
            X[:, div_point:], start_1, feature, dim, X_ori, rng, keep_transform, feature_is_transformer
        )
        intervals.extend(intervals_R)
        if keep_transform and Xt_R.shape[1] > 0:
            Xt = np.hstack((Xt, Xt_R))

        return intervals, Xt

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
    ):
        self.n_estimators = n_estimators
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.n_components = n_components
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
        for t in transforms:
            ui = UnsupervisedIntervals(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                n_jobs=self._n_jobs,
                random_state=self.random_state,
            )
            features = ui.fit_transform(t)
            Xt = np.hstack((Xt, features))
            self._transformers.append(ui)

        self.rp_ = GaussianRandomProjection(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        Xt = self.rp_.fit_transform(Xt)

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
