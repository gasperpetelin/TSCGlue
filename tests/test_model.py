"""Tests for AutoTSC models."""

import pytest
from tscglue.models import LokyStackerV8Base
from sklearn.metrics import accuracy_score
from tscglue import utils


def test_model_accuracy_on_arrowhead():
    """Test model can achieve reasonable accuracy on ArrowHead dataset."""
    X_train, y_train, X_test, y_test = utils.load_dataset("ArrowHead")

    model = LokyStackerV8Base(random_state=270, n_repetitions=1, k_folds=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"

    print(f"Test passed with accuracy: {accuracy:.4f}")


def test_model_on_multivariate():
    """Test model can fit and predict on a multivariate dataset."""
    X_train, y_train, X_test, y_test = utils.load_dataset("BasicMotions")

    model = LokyStackerV8Base(random_state=270, n_repetitions=1, k_folds=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"

    print(f"Test passed with accuracy: {accuracy:.4f}")
