"""Custom transformers for time series classification."""

import numpy as np
from aeon.transformations.collection.base import BaseCollectionTransformer


class CumSum(BaseCollectionTransformer):
    """Cumulative sum transformer with pre-scaling (std) and post-scaling (max=1)."""

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "numpy3D",
        "fit_is_empty": True,
    }

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X with shape (n_instances, n_channels, n_timesteps)."""
        X = np.asarray(X, dtype=float)

        std = X.std(axis=-1, keepdims=True)
        std[std == 0] = 1  # avoid division by zero
        X_scaled = X / std
        X_shifted = X_scaled - X_scaled[..., [0]]
        Xt = np.cumsum(X_shifted, axis=-1)

        max_abs = np.max(np.abs(Xt), axis=-1, keepdims=True)
        max_abs[max_abs == 0] = 1  # avoid division by zero
        Xt = Xt / max_abs

        Xt = Xt - Xt[..., [0]]
        return Xt


class Difference(BaseCollectionTransformer):
    """Difference transformer with configurable lag."""

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "numpy3D",
        "fit_is_empty": True,
    }

    def __init__(self, lag: int = 1) -> None:
        self.lag = lag
        super().__init__()

    def _transform(self, X, y=None):
        """Transform X with shape (n_instances, n_channels, n_timesteps)."""
        if self.lag <= 0:
            raise ValueError(f"lag must be > 0, got {self.lag}")

        Xt = X[:, :, self.lag :] - X[:, :, : -self.lag]
        return Xt


from aeon.transformations.collection.base import BaseCollectionTransformer
from scipy.interpolate import interp1d


class DownsampleTransformer(BaseCollectionTransformer):
    _tags = {
        "X_inner_type": ["np-list", "numpy3D"],
        "capability:multivariate": True,
        "capability:unequal_length": True,
        "fit_is_empty": True,
    }

    def __init__(self, proportion):
        self.proportion = proportion
        super().__init__()

    def _transform(self, X, y=None):
        self._check_parameters()

        is_np = isinstance(X, np.ndarray)
        out = []

        for x in X:
            c, t = x.shape
            new_t = max(2, int(round(t * self.proportion)))

            old_grid = np.linspace(0, 1, t)
            new_grid = np.linspace(0, 1, new_t)

            xr = np.zeros((c, new_t))
            for i in range(c):
                f = interp1d(old_grid, x[i], kind="linear")
                xr[i] = f(new_grid)

            out.append(xr)

        return np.asarray(out) if is_np else out

    def _check_parameters(self):
        if not (0 < self.proportion < 1):
            raise ValueError("proportion must be between 0 and 1")


class PolarCoordinates(BaseCollectionTransformer):
    """
    Convert each channel of a time series into polar coordinates using (x, dx).

    Parameters
    ----------
    lag : int, default=1
        Time lag to compute dx = x(t) - x(t-lag).
    mode : {"both", "magnitude", "angle"}, default="both"
        Selects which polar components to return.
    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "numpy3D",
        "fit_is_empty": True,
    }

    def __init__(self, lag: int = 1, mode: str = "both"):
        if mode not in {"both", "magnitude", "angle"}:
            raise ValueError("mode must be 'both', 'magnitude', or 'angle'")
        self.lag = lag
        self.mode = mode
        super().__init__()

    def _transform(self, X, y=None):
        """
        X shape = (n_instances, n_channels, n_timesteps)
        Returns Xt with:
          - magnitude only: (n_instances, C, T - lag)
          - angle only: (n_instances, C, T - lag)
          - both: (n_instances, 2C, T - lag)
        """
        if self.lag <= 0:
            raise ValueError(f"lag must be > 0, got {self.lag}")

        # x(t)
        x = X[:, :, self.lag :]

        # dx = x(t) - x(t - lag)
        dx = X[:, :, self.lag :] - X[:, :, : -self.lag]

        out_list = []

        # magnitude
        if self.mode in ("both", "magnitude"):
            magnitude = np.sqrt(x * x + dx * dx)
            out_list.append(magnitude)

        # angle
        if self.mode in ("both", "angle"):
            angle = np.arctan2(dx, x + 1e-8)
            out_list.append(angle)

        # concatenate along channel axis
        Xt = np.concatenate(out_list, axis=1)

        return Xt


from scipy.stats import rankdata


class RankTransform(BaseCollectionTransformer):
    """
    Rank-transform each time series independently, per instance and per channel.

    Parameters
    ----------
    method : {"average", "min", "max", "dense", "ordinal"}, default="average"
        Ranking method passed to `scipy.stats.rankdata`.
        Matches behavior of pandas/numpy rank methods.

    normalize : bool, default=True
        If True, normalize ranks to [0, 1].
        If False, return raw ranks in [1, T].
    """

    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "numpy3D",
        "fit_is_empty": True,
    }

    def __init__(self, method="average", normalize=True):
        super().__init__()
        self.method = method
        self.normalize = normalize

    def _transform(self, X, y=None):
        """
        Parameters
        ----------
        X : array of shape (n_instances, n_channels, n_timesteps)

        Returns
        -------
        Xt : array of same shape as X
            Rank-transformed time series.
        """
        n_instances, n_channels, n_t = X.shape
        Xt = np.empty_like(X, dtype=float)

        for i in range(n_instances):
            for c in range(n_channels):
                ranks = rankdata(X[i, c], method=self.method)

                if self.normalize:
                    ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())  # scale to [0, 1]

                Xt[i, c] = ranks

        return Xt


class LocalMeanSubtract(BaseCollectionTransformer):
    _tags = {
        "capability:multivariate": True,
        "X_inner_type": "numpy3D",
        "fit_is_empty": True,
    }

    def __init__(self, k: int | float = 0.1):
        self.k = k
        super().__init__()

    def _transform(self, X, y=None):
        n_instances, n_channels, n_timesteps = X.shape
        Xt = np.empty_like(X, dtype=float)

        for i in range(n_instances):
            for c in range(n_channels):
                series = X[i, c]

                if isinstance(self.k, float) and 0 < self.k < 1:
                    k_val = max(1, int(round(self.k * n_timesteps)))
                else:
                    k_val = int(self.k)
                    if k_val <= 0:
                        raise ValueError(f"k must be positive, got {self.k}")

                csum = np.cumsum(np.r_[0, series])
                out = np.empty_like(series, dtype=float)

                for t in range(n_timesteps):
                    lo = max(0, t - k_val)
                    hi = min(n_timesteps, t + k_val + 1)
                    local_mean = (csum[hi] - csum[lo]) / (hi - lo)
                    out[t] = series[t] - local_mean

                Xt[i, c] = out

        return Xt
