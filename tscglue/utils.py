"""Utility functions for AutoTSC."""

from aeon.datasets import load_classification
from sklearn.model_selection import KFold, StratifiedKFold


def load_dataset(dataset_name):
    """Load and normalize a dataset."""
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")
    return X_train, y_train, X_test, y_test


def get_folds(X, y, n_splits=10, random_state=None):
    folds = []
    try:
        # Try stratified split
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, val_idx in skf.split(X, y):
            folds.append((train_idx.tolist(), val_idx.tolist()))
    except ValueError:
        # Fall back to regular KFold
        print(f"StratifiedKFold failed, falling back to regular KFold with n_splits={n_splits}")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, val_idx in kf.split(X):
            folds.append((train_idx.tolist(), val_idx.tolist()))
    return folds