"""Tests for AutoTSC models."""

import pytest
import numpy as np
from sklearn.metrics import accuracy_score
from autotsc.models import AutoTSCModel
from autotsc.utils import load_dataset


def test_model_accuracy_on_arrowhead():
    """Test that AutoTSCModel2 can achieve reasonable accuracy on ArrowHead dataset."""
    # Load the dataset
    X_train, y_train, X_test, y_test = load_dataset("ArrowHead")

    # Initialize AutoTSCModel2
    model = AutoTSCModel(n_jobs=8, verbose=1)

    # Fit the model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Check that accuracy is within expected range
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"

    print(f"Test passed with accuracy: {accuracy:.4f}")
