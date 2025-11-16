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
