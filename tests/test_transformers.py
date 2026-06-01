"""Tests for all feature transformers (foundation model embeddings, shapelets, intervals)."""

import numpy as np
import pytest
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.transformations.collection.interval_based import QUANTTransformer

from tscglue.interval_models import RSTSFRandomTransformer
from tscglue.models import RDSTFloat64
from tscglue.models_tsfm import Chronos2Embedding, MantisEmbedding


def _X(n_samples=20, n_channels=1, n_timepoints=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_channels, n_timepoints)).astype(np.float32)


def _check(Xt, n_samples):
    assert Xt.ndim == 2
    assert Xt.shape[0] == n_samples
    assert np.isfinite(Xt).all()


TRANSFORMERS = [
    pytest.param(lambda: MultiRocket(n_jobs=1, random_state=0), id="multirocket"),
    pytest.param(lambda: HydraTransformer(n_jobs=1, random_state=0), id="hydra"),
    pytest.param(lambda: QUANTTransformer(), id="quant"),
    pytest.param(lambda: RDSTFloat64(n_jobs=1, random_state=0), id="rdst"),
    pytest.param(
        lambda: RSTSFRandomTransformer(n_intervals=30, random_state=0, n_jobs=1, verbose=False),
        id="rstsf-random",
    ),
    pytest.param(lambda: Chronos2Embedding(include_diff=False), id="chronos2"),
    pytest.param(lambda: MantisEmbedding(include_diff=False), id="mantis"),
]

FM_TRANSFORMERS = [
    pytest.param(lambda: Chronos2Embedding(include_diff=False), id="chronos2"),
    pytest.param(lambda: MantisEmbedding(include_diff=False), id="mantis"),
]


@pytest.mark.parametrize("make_transformer", TRANSFORMERS)
@pytest.mark.parametrize("n_channels", [1, 3])
def test_transformer_shape(make_transformer, n_channels):
    X = _X(n_channels=n_channels)
    t = make_transformer()
    Xt = t.fit_transform(X)
    _check(Xt, X.shape[0])


@pytest.mark.parametrize("make_transformer", FM_TRANSFORMERS)
def test_fm_multivariate_channel_scaling(make_transformer):
    n_channels = 3
    X = _X(n_samples=10, n_channels=n_channels)
    Xt_uni = make_transformer().fit(X[:, :1, :]).transform(X[:, :1, :])
    Xt_multi = make_transformer().fit(X).transform(X)
    _check(Xt_multi, X.shape[0])
    assert Xt_multi.shape[1] == n_channels * Xt_uni.shape[1]


def test_rstsf_random_modes_match():
    X = _X()
    Xt = {}
    for mode in ["fast", "default"]:
        t = RSTSFRandomTransformer(
            n_intervals=30, random_state=42, n_jobs=1, verbose=False, mode=mode
        )
        Xt[mode] = t.fit_transform(X)
    assert Xt["fast"].shape == Xt["default"].shape
    np.testing.assert_allclose(Xt["fast"], Xt["default"], rtol=1e-5, atol=1e-5)
