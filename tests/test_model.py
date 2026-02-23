"""Tests for AutoTSC models."""

import pytest
from tscglue.models import LokyStackerV7
from tscglue.old_models import LokyStackerV5, LokyStackerV6
from sklearn.metrics import accuracy_score
from tscglue import utils


def test_model_accuracy_on_arrowhead():
    """Test model can achieve reasonable accuracy on ArrowHead dataset."""
    X_train, y_train, X_test, y_test = utils.load_dataset("ArrowHead")

    model = LokyStackerV5(random_state=270, n_repetitions=1, k_folds=10, time_limit_in_seconds=None)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"

    print(f"Test passed with accuracy: {accuracy:.4f}")


SMALL_DATASETS = [
    "ItalyPowerDemand",
    "SmoothSubspace",
    "Chinatown",
    "Coffee",
    "BME",
]


@pytest.mark.parametrize("dataset_name", SMALL_DATASETS)
def test_v5_v6_v7_accuracy_match(dataset_name):
    """Test that LokyStackerV5, V6, and V7 produce the same accuracy on small datasets."""
    X_train, y_train, X_test, y_test = utils.load_dataset(dataset_name)
    X_train = X_train.astype("float64")
    X_test = X_test.astype("float64")

    seed = 42
    k_folds = 5

    v5 = LokyStackerV5(random_state=seed, n_repetitions=1, k_folds=k_folds, n_jobs=4)
    v5.fit(X_train, y_train)
    acc_v5 = accuracy_score(y_test, v5.predict(X_test))

    v6 = LokyStackerV6(random_state=seed, n_repetitions=1, k_folds=k_folds, n_jobs=4)
    v6.fit(X_train, y_train)
    acc_v6 = accuracy_score(y_test, v6.predict(X_test))

    v7 = LokyStackerV7(random_state=seed, n_repetitions=1, k_folds=k_folds, n_jobs=4)
    v7.fit(X_train, y_train)
    acc_v7 = accuracy_score(y_test, v7.predict(X_test))

    print(f"{dataset_name}: V5={acc_v5:.4f}, V6={acc_v6:.4f}, V7={acc_v7:.4f}")

    assert acc_v5 == acc_v6, f"{dataset_name}: V5 ({acc_v5:.4f}) != V6 ({acc_v6:.4f})"
    assert acc_v6 == acc_v7, f"{dataset_name}: V6 ({acc_v6:.4f}) != V7 ({acc_v7:.4f})"
