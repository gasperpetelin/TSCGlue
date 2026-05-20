"""Standalone script: run Chronos2Embedding on synthetic data (CPU and GPU)."""

import time
import numpy as np
from tscglue.models_tsfm import Chronos2Embedding

if __name__ == "__main__":
    print("starting...", flush=True)
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((20, 1, 50)).astype(np.float32)
    X_test = rng.standard_normal((8, 1, 50)).astype(np.float32)

    for device in ["cpu", "cuda"]:
        print(f"\n--- device={device} ---", flush=True)
        emb = Chronos2Embedding(include_diff=False, verbose=True, device=device)
        emb.fit(X_train)

        t0 = time.perf_counter()
        Xt_train = emb.transform(X_train)
        Xt_test = emb.transform(X_test)
        elapsed = time.perf_counter() - t0

        print(f"Train embeddings: {Xt_train.shape}  dtype={Xt_train.dtype}")
        print(f"Test  embeddings: {Xt_test.shape}  dtype={Xt_test.dtype}")
        print(f"Finite: train={np.isfinite(Xt_train).all()}  test={np.isfinite(Xt_test).all()}")
        print(f"Time: {elapsed:.2f}s", flush=True)
