"""Standalone script: run TSCGlueClassifier on synthetic data with max verbosity (CPU and GPU)."""

import numpy as np
from sklearn.metrics import accuracy_score
from tscglue.models import TSCGlueClassifier

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((60, 1, 50)).astype(np.float32)
    y_train = np.array(["a"] * 30 + ["b"] * 30)
    X_test = rng.standard_normal((20, 1, 50)).astype(np.float32)
    y_test = np.array(["a"] * 10 + ["b"] * 10)

    for n_gpus in [0, 1]:
        print(f"\n{'='*60}")
        print(f"n_gpus={n_gpus} ({'GPU' if n_gpus else 'CPU'})")
        print(f"{'='*60}")
        model = TSCGlueClassifier(random_state=42, n_repetitions=1, k_folds=3, n_jobs=1, verbose=3, n_gpus=n_gpus)
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f"\nAccuracy: {accuracy:.4f}")
            print("\nSummary:")
            for entry in model.summary(return_transforms=True):
                print(f"  {entry}")
        finally:
            model.cleanup()
