"""Standalone script: run TSCGlueClassifier on ArrowHead with max verbosity."""

from sklearn.metrics import accuracy_score
from tscglue import utils
from tscglue.models import TSCGlueClassifier

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = utils.load_dataset("ArrowHead")

    model = TSCGlueClassifier(random_state=42, n_repetitions=1, k_folds=3, n_jobs=1, verbose=3)
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
