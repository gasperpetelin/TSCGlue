"""Utility functions for AutoTSC."""

import os
import numpy as np
import tensorflow as tf
import polars as pl
from aeon.datasets import load_classification
from sklearn.model_selection import StratifiedKFold
from contextlib import contextmanager
import ray

def load_dataset(dataset_name):
    """Load and normalize a dataset."""
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")
    return X_train, y_train, X_test, y_test


def get_resource_config(n_jobs=-1, n_gpus=-1):
    """
    Detect and configure CPU/GPU resources.

    Args:
        n_jobs: Number of CPUs to use. -1 means use all available.
        n_gpus: Number of GPUs to use. -1 means use all available.

    Returns:
        tuple: (cpus_available, cpus_to_use, gpus_available, gpus_to_use)
    """
    # Detect available CPUs
    n_cpus_available = os.cpu_count() or 1
    if n_jobs == -1:
        n_cpus_to_use = n_cpus_available
    else:
        n_cpus_to_use = min(n_jobs, n_cpus_available)

    # Detect available GPUs
    n_gpus_available = len(tf.config.list_physical_devices('GPU'))
    if n_gpus == -1:
        n_gpus_to_use = n_gpus_available
    else:
        n_gpus_to_use = min(n_gpus, n_gpus_available)

    return n_cpus_available, n_cpus_to_use, n_gpus_available, n_gpus_to_use


def generate_fold_indices(X, y, n_folds=8, shuffle=True):
    """
    Generate stratified fold indices for cross-validation.

    Args:
        X: Input data
        y: Target labels
        n_folds: Number of folds (default: 8)
        shuffle: Whether to shuffle the data (default: True)

    Returns:
        polars.DataFrame: DataFrame with columns 'fold', 'train_idx', 'test_idx'
    """
    folds = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        folds.append({
            'fold': i,
            'train_idx': train_idx,
            'test_idx': test_idx,
        })
    return pl.DataFrame(folds)


def print_fit_start_info(X, y, cpus_to_use, cpus_available, gpus_to_use, gpus_available):
    """
    Print formatted training information.

    Args:
        X: Input data array with shape (n_samples, n_channels, n_timesteps)
        y: Target labels
        cpus_to_use: Number of CPUs to use
        cpus_available: Total CPUs available
        gpus_to_use: Number of GPUs to use
        gpus_available: Total GPUs available
    """
    lines = [
        f"Number of samples: {X.shape[0]}",
        f"Number of channels: {X.shape[1]}",
        f"Length of series: {X.shape[2]}",
        f"Number of classes: {len(np.unique(y))}",
        f"CPUs: {cpus_to_use}/{cpus_available}",
        f"GPUs: {gpus_to_use}/{gpus_available}",
    ]
    max_len = max(len(line) for line in lines)
    border = "|" + "-" * (max_len + 2) + "|"
    print(border)
    for line in lines:
        print(f"| {line.ljust(max_len)} |")
    print(border)



@contextmanager
def ray_init_or_reuse(**ray_init_kwargs):
    started_here = False
    try:
        # If Ray is already running, reuse it
        if not ray.is_initialized():
            # Start Ray with the requested resources only if not running
            ray.init(**ray_init_kwargs)
            started_here = True

        yield

    finally:
        # Only shutdown if we started it
        if started_here and ray.is_initialized():
            ray.shutdown()