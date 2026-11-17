"""Tests for the final heads in tscglue.heads."""

import numpy as np
import pytest

from tscglue import utils
from tscglue.heads import Blend, Caruana, Mean, WMean, blend_candidates


def _members(n=120, n_classes=3, seed=0):
    """One informative member (mostly right, a bit overconfident) and three noise members."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    good = np.full((n, n_classes), 0.02)
    hit = rng.random(n) < 0.8
    good[np.arange(n), np.where(hit, y, (y + 1) % n_classes)] = 0.96
    noise = rng.dirichlet(np.ones(n_classes), size=(3, n))
    return np.concatenate([good[None], noise]), y


def _log_loss(p, y):
    return -np.log(p[np.arange(len(y)), y]).mean()


@pytest.mark.parametrize("head", [Mean(), WMean(), WMean(weights=False), Caruana()])
def test_heads_return_distributions(head):
    P, y = _members()
    p = head.fit(P, y).predict(P)
    assert p.shape == P.shape[1:]
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_wmean_beats_plain_mean():
    P, y = _members()
    assert _log_loss(WMean().fit(P, y).predict(P), y) < _log_loss(Mean().predict(P), y)


def test_wmean_weights_the_informative_member():
    P, y = _members()
    head = WMean().fit(P, y)
    assert np.argmax(head.a_) == 0


def test_caruana_picks_the_informative_member():
    P, y = _members()
    head = Caruana().fit(P, y)
    assert np.argmax(head.weights_) == 0
    np.testing.assert_allclose(head.weights_.sum(), 1.0)


def test_caruana_binary():
    P, y = _members(n_classes=2)
    p = Caruana().fit(P, y).predict(P)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)


def test_blend_fit_predict():
    P, y = _members()
    splits = utils.get_folds(y, y, n_splits=4, random_state=0)
    blend = Blend(blend_candidates([0, 1], [0, 1, 2, 3])).fit(P, y, splits)
    p = blend.predict(P)
    assert p.shape == P.shape[1:]
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-9)
    assert set(blend.weights_) == {"mean-L1", "wmean-T-ll-L1", "wmean-T-ll-L0L1",
                                   "caruana-T-ll-L1", "caruana-T-ll-L0L1"}
    np.testing.assert_allclose(sum(blend.weights_.values()), 1.0)
    assert all(len(heads) == 4 for heads in blend.fold_heads_.values())
