"""Tests for AutoTSC models."""

import pytest
import numpy as np
from tscglue.models import LokyStackerV8Base, LokyStackerV10Base
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


@pytest.mark.parametrize("feature_dtype", [None, "float32", "float64"])
def test_v10base_feature_dtype(feature_dtype):
    """Test that LokyStackerV10Base fit+predict works with different feature_dtype values."""
    X_train, y_train, X_test, y_test = utils.load_dataset("ArrowHead")

    model = LokyStackerV10Base(
        random_state=270, n_repetitions=1, k_folds=10, feature_dtype=feature_dtype,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"

    expected_dtype = np.dtype(feature_dtype) if feature_dtype else X_train.dtype
    assert model.feature_dtype == expected_dtype, (
        f"Expected feature_dtype={expected_dtype}, got {model.feature_dtype}"
    )

    print(f"feature_dtype={feature_dtype}: accuracy={accuracy:.4f}")


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
