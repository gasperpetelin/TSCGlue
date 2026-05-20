"""Standalone script: run Chronos2Embedding on synthetic data (CPU and GPU)."""

import argparse
import time
import numpy as np
from tscglue.models_tsfm import Chronos2Embedding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--series-len", type=int, default=512)
    args = parser.parse_args()

    print(f"starting... n_train={args.n_train} n_test={args.n_test} series_len={args.series_len}", flush=True)
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((args.n_train, 1, args.series_len)).astype(np.float32)
    X_test = rng.standard_normal((args.n_test, 1, args.series_len)).astype(np.float32)

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
