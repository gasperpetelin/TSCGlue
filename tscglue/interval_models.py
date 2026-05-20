"""RSTSF variants: RandomIntervals and UnsupervisedIntervals + GaussianRandomProjection."""

__all__ = ["UnsupervisedIntervals", "RSTSFRandomTransformer", "RSTSFCombinedTransformer"]

import inspect

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.random_projection import GaussianRandomProjection
from sklearn.utils import check_random_state

from aeon.base._base import _clone_estimator
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


class _FastRandomIntervals:
    """Wraps aeon's RandomIntervals fit, replaces transform with vectorized numpy.

    Aeon's _transform_interval does `[[f] for f in feature(seg)]` — a Python
    loop over every sample for every interval. The feature functions already
    accept 2D (n_samples, length) and return 1D, so we just call them directly.

    fit is 100% identical to aeon (same RNG, same deduplication), so intervals_
    and all results are bit-for-bit identical.
    """

    def __init__(self, n_intervals=100, min_interval_length=3, random_state=None):
        self._ri = RandomIntervals(
            n_intervals=n_intervals,
            min_interval_length=min_interval_length,
            random_state=random_state,
            n_jobs=1,
        )

    def fit(self, X, y=None):
        self._ri.fit(X, y)
        return self

    def transform(self, X):
        parts = []
        for start, end, dim, feature, dilation in self._ri.intervals_:
            seg = X[:, dim, start:end:dilation]  # (n_samples, length)
            parts.append(feature(seg).reshape(-1, 1))
        return np.hstack(parts)


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
    from tscglue.utils import RidgeClassifierCVDecisionProba
    return RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10))


class RSTSFRandomTransformer(BaseEstimator, TransformerMixin):
    """Feature extraction stage of RSTSFRandom, usable as a standalone transformer.

    Applies RandomIntervals to the same 4 series representations as RSTSF
    (original, first differences, periodogram, AR coefficients). The result is
    a fixed-width feature matrix that can be fed to any downstream classifier.

    Parameters
    ----------
    n_intervals : int, default=600
    min_interval_length : int, default=3
    mode : str, default="fast"
        Implementation to use for interval extraction.
        "fast"    - vectorised numpy transform (much faster than aeon default).
        "default" - aeon's RandomIntervals unmodified.
    random_state : None, int or RandomState, default=None
    n_jobs : int, default=1
    verbose : bool, default=True
    """

    _REP_NAMES = ["raw", "diff", "periodogram", "ar"]

    def __init__(self, n_intervals=600, min_interval_length=3, mode="fast", random_state=None, n_jobs=1, verbose=False):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.mode = mode
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[RSTSFRandomTransformer] {msg}")

    def _make_ri(self):
        if self.mode == "fast":
            return _FastRandomIntervals(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                random_state=self.random_state,
            )
        elif self.mode == "default":
            return RandomIntervals(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                random_state=self.random_state,
                n_jobs=1,
            )
        else:
            raise ValueError(f"mode must be 'fast' or 'default', got '{self.mode}'")

    def fit(self, X, y=None):
        from time import perf_counter
        self._n_jobs = check_n_jobs(self.n_jobs)

        t0 = perf_counter()
        transforms, self._series_transformers = _make_series_transforms(X)
        self._log(f"series transforms: {perf_counter() - t0:.2f}s")

        self._ri_transformers = []
        for name, t in zip(self._REP_NAMES, transforms):
            t0 = perf_counter()
            ri = self._make_ri()
            ri.fit(t)
            self._ri_transformers.append(ri)
            self._log(f"fit RI  rep={name}: {perf_counter() - t0:.2f}s")
        return self

    def transform(self, X):
        from time import perf_counter
        t0 = perf_counter()
        transforms = [X] + [t.transform(X) for t in self._series_transformers]
        self._log(f"transform series reps: {perf_counter() - t0:.2f}s")

        parts = []
        for name, ri, t in zip(self._REP_NAMES, self._ri_transformers, transforms):
            t0 = perf_counter()
            part = ri.transform(t)
            self._log(f"transform RI  rep={name}: {perf_counter() - t0:.2f}s")
            parts.append(part)
        return np.hstack(parts)


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


