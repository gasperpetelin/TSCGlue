"""DrCIF-style fixed feature extractor.

DrCIF re-draws random intervals for every tree in its forest, so it has no single
feature matrix. This module freezes that randomness into a reusable extractor: a
large pool of random intervals is drawn once over DrCIF's three representations,
and catch22 + 7 summary statistics are computed per interval. The resulting fixed
matrix can be fed to any estimator.

To stay close to DrCIF, pair the extractor with a random-subspace tree ensemble
(ExtraTrees with ``max_features='sqrt'``): its per-split feature subsampling
re-creates DrCIF's per-tree interval/attribute randomisation, drawing from the
extracted pool instead of an unbounded set. See :func:`drcif_like_classifier` /
:func:`drcif_like_regressor`.

The features and representations are imported from aeon's DrCIF implementation so
they match bit-for-bit; this pins compatibility to the installed aeon version.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from aeon.classification.interval_based._drcif import (
    row_iqr,
    row_mean,
    row_median,
    row_numba_max,
    row_numba_min,
    row_slope,
    row_std,
)
from aeon.transformations.collection import PeriodogramTransformer
from aeon.transformations.collection.feature_based import Catch22
from aeon.transformations.collection.interval_based import RandomIntervals
from aeon.utils.numba.general import first_order_differences_3d


class DrCIFExtractor(TransformerMixin, BaseEstimator):
    """Fixed feature extractor mirroring DrCIF's per-tree interval extraction.

    For each of DrCIF's three representations (base series, first-order
    differences, periodogram) a single :class:`RandomIntervals` transformer is
    fitted with DrCIF's feature set (catch22 + mean, std, slope, median, IQR,
    min, max). The three feature blocks are concatenated into one matrix.

    Parameters
    ----------
    n_intervals : int, default=200
        Size of the random interval pool drawn per representation. Larger pools
        give the downstream ensemble more intervals to subsample from, bringing
        the result closer to DrCIF (which draws fresh intervals per tree).
    min_interval_length : int or float, default=3
        Minimum interval length. Float values are a proportion of series length.
    max_interval_length : int or float, default=0.5
        Maximum interval length. Float values are a proportion of series length.
        DrCIF's default is ``0.5``.
    use_pycatch22 : bool, default=False
        Use the C ``pycatch22`` implementation for the catch22 features.
    random_state : int or None, default=None
        Seed for interval sampling. Each representation uses ``random_state + i``
        so their interval pools differ.
    n_jobs : int, default=1
        Threads passed to each :class:`RandomIntervals`.

    Attributes
    ----------
    steps_ : list of (transformer or None, RandomIntervals)
        Fitted representation transformer and interval extractor per block.
    """

    def __init__(
        self,
        n_intervals: int = 200,
        min_interval_length: int | float = 3,
        max_interval_length: int | float = 0.5,
        use_pycatch22: bool = False,
        random_state: int | None = None,
        n_jobs: int = 1,
    ):
        self.n_intervals = n_intervals
        self.min_interval_length = min_interval_length
        self.max_interval_length = max_interval_length
        self.use_pycatch22 = use_pycatch22
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _representations(self):
        """DrCIF's three series representations (``None`` = base series)."""
        return [
            None,
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(),
        ]

    def _features(self):
        """DrCIF's per-interval feature set: catch22 + 7 summary statistics."""
        return [
            Catch22(outlier_norm=True, use_pycatch22=self.use_pycatch22),
            row_mean,
            row_std,
            row_slope,
            row_median,
            row_iqr,
            row_numba_min,
            row_numba_max,
        ]

    def fit(self, X, y=None):
        self.steps_ = []
        for i, rep in enumerate(self._representations()):
            Xr = rep.fit_transform(X) if rep is not None else X
            ri = RandomIntervals(
                n_intervals=self.n_intervals,
                min_interval_length=self.min_interval_length,
                max_interval_length=self.max_interval_length,
                features=self._features(),
                random_state=None if self.random_state is None else self.random_state + i,
                n_jobs=self.n_jobs,
            )
            ri.fit(Xr, y)
            self.steps_.append((rep, ri))
        return self

    def transform(self, X):
        blocks = []
        for rep, ri in self.steps_:
            Xr = rep.transform(X) if rep is not None else X
            blocks.append(np.nan_to_num(ri.transform(Xr)))
        return np.hstack(blocks)


def drcif_like_classifier(
    random_state: int | None = None,
    n_jobs: int = 1,
    n_intervals: int = 200,
    n_estimators: int = 200,
) -> Pipeline:
    """DrCIF-like classifier: :class:`DrCIFExtractor` + ExtraTrees.

    ``max_features='sqrt'`` makes the ensemble subsample features per split,
    emulating DrCIF's per-tree interval/attribute randomisation.
    """
    return Pipeline(
        [
            (
                "features",
                DrCIFExtractor(
                    n_intervals=n_intervals, random_state=random_state, n_jobs=n_jobs
                ),
            ),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    max_features="sqrt",
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def drcif_like_regressor(
    random_state: int | None = None,
    n_jobs: int = 1,
    n_intervals: int = 200,
    n_estimators: int = 200,
) -> Pipeline:
    """DrCIF-like regressor: :class:`DrCIFExtractor` + ExtraTrees regressor."""
    return Pipeline(
        [
            (
                "features",
                DrCIFExtractor(
                    n_intervals=n_intervals, random_state=random_state, n_jobs=n_jobs
                ),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    max_features="sqrt",
                    random_state=random_state,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )
