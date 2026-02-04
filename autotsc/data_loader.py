"""Load datasets from the local data/ directory."""

import os

from aeon.datasets import load_from_ts_file

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_fold(dataset_name: str, fold: int):
    dataset_dir = os.path.join(DATA_DIR, dataset_name)

    train_path = os.path.join(dataset_dir, f"{dataset_name}{fold}_TRAIN.ts")
    test_path = os.path.join(dataset_dir, f"{dataset_name}{fold}_TEST.ts")

    X_train, y_train = load_from_ts_file(train_path)
    X_test, y_test = load_from_ts_file(test_path)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Example usage
    X_train, y_train, X_test, y_test = load_fold("Crop", fold=0)
    print(f"Loaded Crop fold 0: {X_train.shape}, {y_train.shape}, {X_test.shape}, {y_test.shape}")
