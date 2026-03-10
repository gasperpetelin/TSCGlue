import os
import re
import random
from itertools import product
from pathlib import Path
from urllib.parse import urlparse

import boto3
import click
import numpy as np
import polars as pl
from botocore.exceptions import ClientError
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.dummy import DummyClassifier
from aeon.classification.feature_based import Catch22Classifier
from tscglue.data_loader import DATA_DIR, load_fold
from tscglue.models_tsfm import Chronos2Classifier, ALL_TSFM_MODELS, make_tsfm_model
from tscglue.gpu_models import MRHydraClassifier, MultiRocketHydraSelectKBestClassifier
from tscglue.models import (
    LokyStackerV7,
    LokyStackerV7SoftET,
    LokyStackerV9Base,
    LokyStackerV7SoftFilterRidge,
    LokyStackerV7SoftRidge,
    LokyStackerV7SoftRF,
    LokyStackerV8Base,
    LokyStackerV8AutoBestStacking,
    LokyStackerV8AutoBestBase,
    LokyStackerV8AutoBest,
    LokyStackerV10Base,
    LokyStackerV7Filter_M,
    LokyStackerV7Filter_Q,
    LokyStackerV7Filter_R,
    LokyStackerV7Filter_S,
    LokyStackerV7Filter_MQ,
    LokyStackerV7Filter_MR,
    LokyStackerV7Filter_MS,
    LokyStackerV7Filter_QR,
    LokyStackerV7Filter_QS,
    LokyStackerV7Filter_RS,
    LokyStackerV7Filter_MQR,
    LokyStackerV7Filter_MQS,
    LokyStackerV7Filter_MRS,
    LokyStackerV7Filter_QRS,
    LokyStackerV7Filter_MQRS,
    TSCGlue,
)

_FILTER_VARIANTS = {
    "loky-filter-M":    LokyStackerV7Filter_M,
    "loky-filter-Q":    LokyStackerV7Filter_Q,
    "loky-filter-R":    LokyStackerV7Filter_R,
    "loky-filter-S":    LokyStackerV7Filter_S,
    "loky-filter-MQ":   LokyStackerV7Filter_MQ,
    "loky-filter-MR":   LokyStackerV7Filter_MR,
    "loky-filter-MS":   LokyStackerV7Filter_MS,
    "loky-filter-QR":   LokyStackerV7Filter_QR,
    "loky-filter-QS":   LokyStackerV7Filter_QS,
    "loky-filter-RS":   LokyStackerV7Filter_RS,
    "loky-filter-MQR":  LokyStackerV7Filter_MQR,
    "loky-filter-MQS":  LokyStackerV7Filter_MQS,
    "loky-filter-MRS":  LokyStackerV7Filter_MRS,
    "loky-filter-QRS":  LokyStackerV7Filter_QRS,
    "loky-filter-MQRS": LokyStackerV7Filter_MQRS,
}

import boto3
from botocore.exceptions import ClientError
from urllib.parse import urlparse


class S3FileCache:
    def __init__(self, base_s3_dir: str):
        self.base_s3_dir = base_s3_dir.rstrip("/")
        parsed = urlparse(base_s3_dir)
        self.bucket = parsed.netloc
        self.prefix = parsed.path.lstrip("/").rstrip("/")
        self._s3 = boto3.client("s3")

        self._files = set()
        self._loaded = False

    def _full_key(self, filename: str) -> str:
        return f"{self.prefix}/{filename}" if self.prefix else filename

    def _load_once(self):
        if self._loaded:
            return

        paginator = self._s3.get_paginator("list_objects_v2")

        for page in paginator.paginate(
            Bucket=self.bucket,
            Prefix=self.prefix
        ):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                # store only filename part
                filename = key[len(self.prefix) + 1:] if self.prefix else key
                self._files.add(filename)

        self._loaded = True

    def exists(self, filename: str) -> bool:
        self._load_once()

        if filename in self._files:
            return True

        key = self._full_key(filename)

        try:
            self._s3.head_object(Bucket=self.bucket, Key=key)
            self._files.add(filename)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def add(self, df: pl.DataFrame, filename: str):
        full_name = f"{self.base_s3_dir}/{filename}"
        df.write_parquet(full_name)
        self._files.add(filename)


class LocalFileCache:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def exists(self, filename: str) -> bool:
        return (self.base_dir / filename).exists()

    def add(self, df: pl.DataFrame, filename: str):
        df.write_parquet(self.base_dir / filename)


def optimal_k(n_train, k_min=6000, k_max=35000, midpoint=300, steepness=0.010):
    return int(k_min + (k_max - k_min) / (1 + np.exp(-steepness * (n_train - midpoint))))


def get_model(model_name, random_state, n_train=None, n_jobs=8):
    if model_name == "mr-hydra-kbest-auto":
        if n_train is None:
            raise ValueError("n_train is required for mr-hydra-kbest-auto")
        k = optimal_k(n_train)
        e = Pipeline([
            ("var", VarianceThreshold()),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
        ])
        return MRHydraClassifier(estimator=e, n_jobs=n_jobs, random_state=random_state)
    elif model_name == "mr-hydra-contained-auto":
        return MultiRocketHydraSelectKBestClassifier(k=None, n_jobs=n_jobs, random_state=random_state)
    elif model_name == "loky-stacker-v7":
        return LokyStackerV7(random_state=random_state, n_repetitions=1, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v7-soft-et":
        return LokyStackerV7SoftET(random_state=random_state, n_repetitions=1, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v7-soft-ridge":
        return LokyStackerV7SoftRidge(random_state=random_state, n_repetitions=1, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v7-soft-rf":
        return LokyStackerV7SoftRF(random_state=random_state, n_repetitions=1, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v7-soft-filter-ridge":
        return LokyStackerV7SoftFilterRidge(random_state=random_state, n_repetitions=1, n_jobs=n_jobs, verbose=10)
    elif model_name in ("loky-stacker-v8-base", "loky-stacker-v8-base-r1"):
        return LokyStackerV8Base(random_state=random_state, n_repetitions=1, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v8-base-r3":
        return LokyStackerV8Base(random_state=random_state, n_repetitions=3, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v8-auto-best-stacking":
        return LokyStackerV8AutoBestStacking(random_state=random_state, n_repetitions=1, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v8-auto-best-base":
        return LokyStackerV8AutoBestBase(random_state=random_state, n_repetitions=1, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v8-auto-best":
        return LokyStackerV8AutoBest(random_state=random_state, n_repetitions=1, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v9-base-r1":
        return LokyStackerV9Base(random_state=random_state, n_repetitions=1, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v9-base-r2":
        return LokyStackerV9Base(random_state=random_state, n_repetitions=2, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v9-base-r3":
        return LokyStackerV9Base(random_state=random_state, n_repetitions=3, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v9-base-r5":
        return LokyStackerV9Base(random_state=random_state, n_repetitions=5, n_jobs=n_jobs, verbose=10)
    elif model_name == "chronos2":
        return Chronos2Classifier()
    elif model_name == "mydummy":
        return DummyClassifier()
    elif model_name == "mycatch22":
        return Catch22Classifier(random_state=random_state)
    elif model_name == "TSCGlue-3-3-26":
        return TSCGlue(random_state=random_state, n_jobs=n_jobs)
    elif model_name == "mycatch22v2":
        return Catch22Classifier(random_state=random_state + 1000)
    elif model_name == "mymrhydra":
        return MultiRocketHydraClassifier(random_state=random_state, n_jobs=n_jobs)
    elif model_name == "mymrhydrav2":
        return MultiRocketHydraClassifier(random_state=random_state + 1000, n_jobs=n_jobs)
    elif model_name == "loky-stacker-v10-base":
        return LokyStackerV10Base(random_state=random_state, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v10-base-2x":
        _base = ["multirockethydra-bestk-p-ridgecv", "quant-etc", "rdst-p-ridgecv", "rstsf"]
        return LokyStackerV10Base(random_state=random_state, n_jobs=n_jobs, verbose=10, model_names=_base * 2)
    elif model_name == "loky-stacker-v10-base-5x":
        _base = ["multirockethydra-bestk-p-ridgecv", "quant-etc", "rdst-p-ridgecv", "rstsf"]
        return LokyStackerV10Base(random_state=random_state, n_jobs=n_jobs, verbose=10, model_names=_base * 5)
    elif model_name == "loky-stacker-v10-base-r3":
        return LokyStackerV10Base(random_state=random_state, n_jobs=n_jobs, verbose=10, n_repetitions=3)
    elif model_name in _FILTER_VARIANTS:
        return _FILTER_VARIANTS[model_name](random_state=random_state, n_repetitions=1, n_jobs=n_jobs, verbose=10)
    elif model_name == "mantis-ridgecv":
        return make_tsfm_model("mantis-ridgecv", random_state=random_state)
    elif model_name == "mantis-rf":
        return make_tsfm_model("mantis-rf", random_state=random_state)
    elif model_name == "mantis-et":
        return make_tsfm_model("mantis-et", random_state=random_state)
    elif model_name == "mantis-hgb":
        return make_tsfm_model("mantis-hgb", random_state=random_state)
    elif model_name == "mantis-lgbm":
        return make_tsfm_model("mantis-lgbm", random_state=random_state)
    elif model_name == "chronos2-ridgecv":
        return make_tsfm_model("chronos2-ridgecv", random_state=random_state)
    elif model_name == "chronos2-rf":
        return make_tsfm_model("chronos2-rf", random_state=random_state)
    elif model_name == "chronos2-et":
        return make_tsfm_model("chronos2-et", random_state=random_state)
    elif model_name == "chronos2-hgb":
        return make_tsfm_model("chronos2-hgb", random_state=random_state)
    elif model_name == "chronos2-lgbm":
        return make_tsfm_model("chronos2-lgbm", random_state=random_state)
    elif model_name == "mantis+chronos2-ridgecv":
        return make_tsfm_model("mantis+chronos2-ridgecv", random_state=random_state)
    elif model_name == "mantis+chronos2-rf":
        return make_tsfm_model("mantis+chronos2-rf", random_state=random_state)
    elif model_name == "mantis+chronos2-et":
        return make_tsfm_model("mantis+chronos2-et", random_state=random_state)
    elif model_name == "mantis+chronos2-hgb":
        return make_tsfm_model("mantis+chronos2-hgb", random_state=random_state)
    elif model_name == "mantis+chronos2-lgbm":
        return make_tsfm_model("mantis+chronos2-lgbm", random_state=random_state)
    elif model_name.startswith("mr-hydra-kbest-"):
        k = int(model_name.split("-")[-1])
        e = Pipeline([
            ("var", VarianceThreshold()),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))),
        ])
        return MRHydraClassifier(estimator=e, n_jobs=n_jobs, random_state=random_state)
    else:
        raise ValueError(f"Unknown model name: {model_name}")


ALL_MODELS = [
    # "loky-stacker-v5-r1",
    # "loky-stacker-v5-soft-et",
    # "loky-stacker-v5-soft-ridge",
    # "loky-stacker-v5-soft-rf",
    "loky-stacker-v6",
    "loky-stacker-v6-soft-et",
    "loky-stacker-v6-soft-ridge",
    "loky-stacker-v6-soft-rf",
    "mr-hydra-kbest-5000",
    "mr-hydra-kbest-10000",
    "mr-hydra-kbest-30000",
    "mr-hydra-kbest-auto",
    "mr-hydra-contained-auto",
    "loky-stacker-v7",
    "loky-stacker-v7-soft-filter-ridge",
    "loky-stacker-v8-base-r1",
    "loky-stacker-v8-base-r3",
    "loky-stacker-v8-auto-best-stacking",
    "loky-stacker-v8-auto-best-base",
    "loky-stacker-v8-auto-best",
    "loky-stacker-v9-base-r1",
    "loky-stacker-v9-base-r2",
    "loky-stacker-v9-base-r3",
    "loky-stacker-v9-base-r5",
    "loky-stacker-v10-base",
    "loky-stacker-v10-base-2x",
    "loky-stacker-v10-base-5x",
    "loky-stacker-v10-base-r3",
    "loky-stacker-v7-soft-et",
    "loky-stacker-v7-soft-ridge",
    "loky-stacker-v7-soft-rf",
    "chronos2",
    "mantis-ridgecv",
    "mantis-rf",
    "mantis-et",
    "mantis-hgb",
    "mantis-lgbm",
    "chronos2-ridgecv",
    "chronos2-rf",
    "chronos2-et",
    "chronos2-hgb",
    "chronos2-lgbm",
    "mantis+chronos2-ridgecv",
    "mantis+chronos2-rf",
    "mantis+chronos2-et",
    "mantis+chronos2-hgb",
    "mantis+chronos2-lgbm",
    "mydummy",
    "mycatch22",
    "TSCGlue-3-3-26",
    "mycatch22v2",
    "mymrhydra",
    "mymrhydrav2",
    #*_FILTER_VARIANTS,
]


def discover_datasets():
    """Return sorted list of dataset names found in data/."""
    return sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )


def discover_folds(dataset_name: str) -> list[int]:
    """Return sorted list of fold numbers available for a dataset."""
    dataset_dir = os.path.join(DATA_DIR, dataset_name)
    pattern = re.compile(rf"^{re.escape(dataset_name)}(\d+)_TRAIN\.ts$")
    folds = []
    for fname in os.listdir(dataset_dir):
        m = pattern.match(fname)
        if m:
            folds.append(int(m.group(1)))
    return sorted(folds)


@click.command()
@click.option("-m", "--models", multiple=True, help="Models to run (can be specified multiple times or comma-separated)")
@click.option("-d", "--datasets", "dataset_names", multiple=True, help="Datasets to run (can be specified multiple times or comma-separated)")
@click.option("-f", "--folds", "fold_spec", default=None, help="Folds to run (comma-separated, e.g. '0,1,2'). Default: all available folds.")
@click.option("-l", "--list-models", is_flag=True, help="List all available models and exit")
@click.option("--list-datasets", is_flag=True, help="List all available datasets and exit")
@click.option("--storage", type=click.Choice(["s3", "disk"]), default="s3", help="Storage backend: s3 or disk")
@click.option("-j", "--n-jobs", default=8, type=int, help="Number of parallel jobs")
def main(models, dataset_names, fold_spec, list_models, list_datasets, storage, n_jobs):
    """Run loky stacking experiments on local fold datasets."""
    all_datasets = discover_datasets()

    if list_models:
        click.echo("Available models:")
        for model in ALL_MODELS:
            click.echo(f"  - {model}")
        return

    if list_datasets:
        click.echo("Available datasets:")
        for ds in all_datasets:
            folds = discover_folds(ds)
            click.echo(f"  - {ds} ({len(folds)} folds)")
        return

    # Determine which models to run
    if models:
        model_list = []
        for m in models:
            model_list.extend([x.strip() for x in m.split(",")])
        invalid_models = [m for m in model_list if m not in ALL_MODELS]
        if invalid_models:
            click.echo(f"Error: Unknown models: {', '.join(invalid_models)}", err=True)
            click.echo("Use -l to list available models", err=True)
            raise click.Abort()
        model_names = model_list
    else:
        model_names = ALL_MODELS
    click.echo(f"Running models: {', '.join(model_names)}")

    # Determine which datasets to run
    if dataset_names:
        dataset_list = []
        for d in dataset_names:
            dataset_list.extend([x.strip() for x in d.split(",")])
        invalid_datasets = [d for d in dataset_list if d not in all_datasets]
        if invalid_datasets:
            click.echo(f"Error: Unknown datasets: {', '.join(invalid_datasets)}", err=True)
            click.echo("Use --list-datasets to list available datasets", err=True)
            raise click.Abort()
        datasets = dataset_list
    else:
        datasets = all_datasets
    click.echo(f"Running datasets: {', '.join(datasets)}")

    # Parse fold spec
    requested_folds = None
    if fold_spec is not None:
        requested_folds = [int(x.strip()) for x in fold_spec.split(",")]

    if storage == "s3":
        cache = S3FileCache("s3://tsc-glue/performance-benchmarking")
    else:
        cache = LocalFileCache("performance-benchmarking")

    # Build all (dataset, model, fold) combos
    combos = []
    for dataset in datasets:
        folds = requested_folds if requested_folds is not None else list(range(30))
        for model_name, fold in product(model_names, folds):
            combos.append((dataset, model_name, fold))

    random.shuffle(combos)

    n = len(combos)
    click.echo(f"Total combinations: {n}")

    for k, (dataset, model_name, fold) in enumerate(combos, 1):
        try:
            stats = {
                "dataset": dataset,
                "model": model_name,
                "fold": fold,
            }

            hash_val = pl.DataFrame([stats]).hash_rows(seed=42, seed_1=1, seed_2=2, seed_3=3).item()

            file_name = f"{hash_val}.parquet"
            if cache.exists(file_name):
                print(f"[{k}/{n}] Skipping: Dataset={dataset}, Fold={fold}, Model={model_name}")
                continue
            else:
                print(f"[{k}/{n}] Processing: Dataset={dataset}, Fold={fold}, Model={model_name}")

            X_train, y_train, X_test, y_test = load_fold(dataset, fold)

            model = get_model(model_name, random_state=fold, n_train=len(X_train), n_jobs=n_jobs)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            if hasattr(model, "cleanup"):
                model.cleanup()
            acc = accuracy_score(y_test, preds)

            stats["test_accuracy"] = acc

            df_stat = pl.DataFrame([stats])
            cache.add(df_stat, file_name)
        except Exception as e:
            print(f"Error processing Dataset={dataset}, Fold={fold}, Model={model_name}: {e}")


if __name__ == "__main__":
    main()
