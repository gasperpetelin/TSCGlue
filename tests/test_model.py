"""Tests for AutoTSC models."""

import os

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
from sklearn.metrics import accuracy_score

from autotsc import utils
from autotsc.models2 import StackerV4Ray


def test_model_accuracy_on_arrowhead():
    """Test that AutoTSCModel2 can achieve reasonable accuracy on ArrowHead dataset."""
    # Load the dataset
    X_train, y_train, X_test, y_test = utils.load_dataset("ArrowHead")

    model = StackerV4Ray(random_state=270, n_repetitions=1, k_folds=10, time_limit_in_seconds=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"

    print(f"Test passed with accuracy: {accuracy:.4f}")
