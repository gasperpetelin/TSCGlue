"""Utility functions for AutoTSC."""

import multiprocessing
import os
import pickle
from time import perf_counter

import numpy as np
import polars as pl
from aeon.datasets import load_classification, load_from_ts_file
from huggingface_hub import hf_hub_download
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import label_binarize

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

#: Metrics :func:`score_predictions` accepts. All but ``log_loss`` are higher-is-better.
EVAL_METRICS = ("accuracy", "f1", "log_loss", "roc_auc", "average_precision")


def log(message: str, level: int, verbose: int, start_time=None, current_time=None):
    if verbose >= level:
        if start_time is not None:
            if current_time is None:
                current_time = perf_counter()
            print(f"[{current_time - start_time:.2f}s] {message}")
        else:
            print(message)


def score_predictions(y_true, proba, classes, metric: str) -> float:
    """Score probability predictions against ``y_true``.

    ``proba`` has one column per entry of ``classes``, in that order. Label-based
    metrics take the argmax and compare as strings so mixed label dtypes still match.
    Multiclass ``roc_auc`` and ``average_precision`` are macro-averaged one-vs-rest.
    """
    if metric == "accuracy":
        preds = np.asarray(classes)[np.argmax(proba, axis=1)]
        return float(accuracy_score(np.asarray(y_true, dtype=str), np.asarray(preds, dtype=str)))
    if metric == "f1":
        preds = np.asarray(classes)[np.argmax(proba, axis=1)]
        return float(
            f1_score(
                np.asarray(y_true, dtype=str),
                np.asarray(preds, dtype=str),
                average="macro",
            )
        )
    if metric == "log_loss":
        return float(log_loss(y_true, proba, labels=classes))
    if metric == "roc_auc":
        if len(classes) == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=classes))
    if metric == "average_precision":
        if len(classes) == 2:
            return float(average_precision_score(y_true, proba[:, 1]))
        return float(
            average_precision_score(label_binarize(y_true, classes=classes), proba, average="macro")
        )
    raise ValueError(f"Unknown eval_metric: {metric!r}")


def save_array(X, name, directory, dtype=None, repetition=None):
    if dtype is not None:
        X = np.asarray(X, dtype=dtype)
    suffix = f"_r_{repetition}" if repetition is not None else ""
    path = f"{directory}/{name}{suffix}.npy"
    np.save(path, X)
    size = os.path.getsize(path) / (1024 * 1024)
    return path, X.shape, size


def read_array(name, directory, repetition=None, mmap_mode="r", allow_pickle=False):
    suffix = f"_r_{repetition}" if repetition is not None else ""
    path = f"{directory}/{name}{suffix}.npy"
    # TODO remove allow_pickle=True once all features/models are numpy arrays
    return np.load(path, mmap_mode=mmap_mode, allow_pickle=allow_pickle)


def save_model(model, name, directory, repetition=None, fold=None):
    rep_suffix = f"_r_{repetition}" if repetition is not None else ""
    fold_suffix = f"_f_{fold}" if fold is not None else ""
    path = f"{directory}/{name}{rep_suffix}{fold_suffix}.pkl"
    os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size = os.path.getsize(path)
    return path, size


def read_model(name, directory, repetition=None, fold=None):
    directory = str(directory)
    rep_suffix = f"_r_{repetition}" if repetition is not None else ""
    fold_suffix = f"_f_{fold}" if fold is not None else ""
    path = f"{directory}/{name}{rep_suffix}{fold_suffix}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _noop():
    return None


def _run_in_subprocess(target, args):
    """Run target(*args) in a fresh spawned subprocess.

    Uses 'spawn' context to avoid inheriting state (e.g. GPU memory, open file
    descriptors, TensorFlow/PyTorch globals) from the parent process, which can
    cause hangs or incorrect behaviour with foundation-model transformers.
    """
    mp_ctx = multiprocessing.get_context("forkserver")
    p = mp_ctx.Process(target=target, args=args)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Subprocess failed with exit code {p.exitcode}")


def require_torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise ImportError(
            "This feature requires PyTorch. Install with:\n"
            "  pip install 'tscglue[torch]'\n"
            "  uv pip install 'tscglue[cpu]'\n"
            "  uv pip install 'tscglue[cu124]'\n"
            "  uv pip install 'tscglue[cu132]'"
        ) from exc


def load_dataset(dataset_name):
    """Load and normalize a dataset."""
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")
    return X_train, y_train, X_test, y_test


def get_folds(X, y, n_splits=10, random_state=None, stratify=True):
    folds = []
    if stratify:
        try:
            # Try stratified split (classification)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for train_idx, val_idx in skf.split(X, y):
                folds.append((train_idx.tolist(), val_idx.tolist()))
            return folds
        except ValueError:
            # Fall back to regular KFold
            print(f"StratifiedKFold failed, falling back to regular KFold with n_splits={n_splits}")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in kf.split(X):
        folds.append((train_idx.tolist(), val_idx.tolist()))
    return folds


def load_ucr_fold(dataset_name: str, fold: int):
    dataset_dir = os.path.join(DATA_DIR, dataset_name)

    train_path = os.path.join(dataset_dir, f"{dataset_name}{fold}_TRAIN.ts")
    test_path = os.path.join(dataset_dir, f"{dataset_name}{fold}_TEST.ts")

    X_train, y_train = load_from_ts_file(train_path)
    X_test, y_test = load_from_ts_file(test_path)

    return X_train, y_train, X_test, y_test


def load_fold(dataset_spec: str, fold: int):
    import re

    # Check for subset suffix like -0.05 or -0.2
    match = re.match(r"^(.+)-(\d+\.?\d*)$", dataset_spec)
    subset_fraction = None
    if match:
        base = match.group(1)
        subset_fraction = float(match.group(2))
        dataset_spec = base

    if dataset_spec.startswith("m-"):
        dataset_name = dataset_spec[2:]
        if subset_fraction is not None:
            return load_fold_monash_subset(dataset_name, fold, subset_fraction)
        return load_fold_monash(dataset_name, fold)
    else:
        if subset_fraction is not None:
            raise ValueError("Subsetting is only supported for monash datasets")
        return load_ucr_fold(dataset_spec, fold)


def load_fold_monash(dataset: str, fold: int = 0):
    repo_id = f"monster-monash/{dataset}"

    path_x = hf_hub_download(repo_id=repo_id, filename=f"{dataset}_X.npy", repo_type="dataset")
    path_y = hf_hub_download(repo_id=repo_id, filename=f"{dataset}_y.npy", repo_type="dataset")
    path_fold = hf_hub_download(
        repo_id=repo_id, filename=f"test_indices_fold_{fold}.txt", repo_type="dataset"
    )

    X = np.load(path_x)
    y = np.load(path_y)
    test_indices = (
        pl.scan_csv(path_fold, has_header=False)
        .with_columns(pl.col("column_1").cast(pl.Int32))
        .collect()["column_1"]
        .to_numpy()
    )

    train_mask = np.ones(len(y), dtype=bool)
    train_mask[test_indices] = False

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_indices], y[test_indices]

    return X_train, y_train, X_test, y_test


def load_fold_monash_subset(dataset: str, fold: int = 0, subset_fraction_to_keep: float = 0.1):
    X_train, y_train, X_test, y_test = load_fold_monash(dataset, fold)
    n = len(y_train)
    keep = max(1, int(n * subset_fraction_to_keep))
    rng = np.random.default_rng(seed=fold)
    idx = rng.choice(n, size=keep, replace=False)
    idx.sort()
    return X_train[idx], y_train[idx], X_test, y_test
