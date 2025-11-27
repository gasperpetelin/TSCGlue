"""Tests for AutoTSC models."""

from sklearn.metrics import accuracy_score

from autotsc import utils
from autotsc.models import AutoTSCModel


def test_model_accuracy_on_arrowhead():
    """Test that AutoTSCModel2 can achieve reasonable accuracy on ArrowHead dataset."""
    # Load the dataset
    X_train, y_train, X_test, y_test = utils.load_dataset("ArrowHead")

    with utils.ray_init_or_reuse(num_cpus=24, resources={"meta": 100}, ignore_reinit_error=True):
        # Initialize AutoTSCModel2
        model = AutoTSCModel(n_jobs=8, verbose=1, model_selection="fast")

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
