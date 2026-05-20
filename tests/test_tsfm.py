"""Tests for foundation model embeddings (Chronos2, Mantis)."""

import numpy as np
import pytest
from tscglue.models_tsfm import Chronos2Embedding, MantisEmbedding
from tscglue import utils


def test_chronos2_embedding_shape():
    X_train, y_train, X_test, y_test = utils.load_dataset("Coffee")
    emb = Chronos2Embedding(include_diff=False)
    emb.fit(X_train)
    Xt_train = emb.transform(X_train)
    Xt_test = emb.transform(X_test)

    assert Xt_train.ndim == 2
    assert Xt_train.shape[0] == X_train.shape[0]
    assert Xt_test.shape[0] == X_test.shape[0]
    assert Xt_train.shape[1] == Xt_test.shape[1]
    assert np.isfinite(Xt_train).all()
    assert np.isfinite(Xt_test).all()


def test_mantis_embedding_shape():
    X_train, y_train, X_test, y_test = utils.load_dataset("Coffee")
    emb = MantisEmbedding(include_diff=False)
    emb.fit(X_train)
    Xt_train = emb.transform(X_train)
    Xt_test = emb.transform(X_test)

    assert Xt_train.ndim == 2
    assert Xt_train.shape[0] == X_train.shape[0]
    assert Xt_test.shape[0] == X_test.shape[0]
    assert Xt_train.shape[1] == Xt_test.shape[1]
    assert np.isfinite(Xt_train).all()
    assert np.isfinite(Xt_test).all()
