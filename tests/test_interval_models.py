"""Tests for interval-based models (RSTSF variants)."""

import numpy as np
import pytest
from tscglue.interval_models import RSTSFRandomTransformer


@pytest.mark.parametrize("mode", ["fast", "default"])
def test_rstsf_random_transformer_modes(mode):
    """Both modes should produce identical output shapes and close feature values."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 1, 50)).astype(np.float32)

    t = RSTSFRandomTransformer(n_intervals=30, random_state=42, n_jobs=1, verbose=False, mode=mode)
    Xt_train = t.fit_transform(X)
    Xt_test = t.transform(X)

    assert Xt_train.ndim == 2
    assert Xt_train.shape[0] == X.shape[0]
    assert Xt_test.shape == Xt_train.shape


def test_rstsf_random_transformer_modes_match():
    """'fast' and 'default' modes must produce numerically identical features."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 1, 50)).astype(np.float32)

    Xt = {}
    for mode in ["fast", "default"]:
        t = RSTSFRandomTransformer(n_intervals=30, random_state=42, n_jobs=1, verbose=False, mode=mode)
        Xt[mode] = t.fit_transform(X)

    assert Xt["fast"].shape == Xt["default"].shape, "Shape mismatch between modes"
    np.testing.assert_allclose(Xt["fast"], Xt["default"], rtol=1e-5, atol=1e-5,
                               err_msg="'fast' and 'default' modes produce different features")
