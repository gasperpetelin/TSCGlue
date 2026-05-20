"""Standalone script: run Chronos2Embedding on synthetic data."""

import numpy as np
from tscglue.models_tsfm import Chronos2Embedding

if __name__ == "__main__":
    print("starting...", flush=True)
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((20, 1, 50)).astype(np.float32)
    X_test = rng.standard_normal((8, 1, 50)).astype(np.float32)

    emb = Chronos2Embedding(include_diff=False, verbose=True)
    emb.fit(X_train)

    Xt_train = emb.transform(X_train)
    Xt_test = emb.transform(X_test)

    print(f"Train embeddings: {Xt_train.shape}  dtype={Xt_train.dtype}")
    print(f"Test  embeddings: {Xt_test.shape}  dtype={Xt_test.dtype}")
    print(f"Finite: train={np.isfinite(Xt_train).all()}  test={np.isfinite(Xt_test).all()}")
