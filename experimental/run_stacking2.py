import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import re
import random
from itertools import product
from pathlib import Path

import click
import numpy as np
import polars as pl
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.dummy import DummyClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.hybrid import HIVECOTEV2
from tscglue.data_loader import DATA_DIR, load_fold
from tscglue.models_tsfm import Chronos2Classifier, ALL_TSFM_MODELS, make_tsfm_model, TabICLTimeSeriesClassifier
from tscglue.gpu_models import MRHydraClassifier, MultiRocketHydraSelectKBestClassifier
from tscglue.interval_models import RSTSFRandom, RSTSFUnsupervised, RSTSFCombined, RSTSFUnsupervisedRaw
from tscglue.models_tsfm import RidgeClassifierCVDecisionProba
from tscglue.models import (
    LokyStackerV10Base,
    LokyStackerV10FM,
    LokyStackerV10FMTSFresh,
    LokyStackerV10RSTSFRandom,
    LokyStackerV10RSTSFRandomMultiStack,
    LokyStackerV10TabICL,
    TSCGlue,
    make_ablation_model,
)


from tscglue.utils import S3FileCache, LocalFileCache


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
    elif model_name == "chronos2":
        return Chronos2Classifier()
    elif model_name == "mydummy":
        return DummyClassifier()
    elif model_name == "mycatch22":
        return Catch22Classifier(random_state=random_state)
    elif model_name == "TSCGlue-3-3-26":
        return TSCGlue(random_state=random_state, n_jobs=n_jobs)
    elif model_name == "TSCGlue-17-4-26":
        return TSCGlue(random_state=random_state, n_jobs=n_jobs)
    elif model_name == "multistack-best-stacking":
        return LokyStackerV10RSTSFRandomMultiStack(random_state=random_state, n_jobs=n_jobs, selection="best-stacking")
    elif model_name == "multistack-best-base":
        return LokyStackerV10RSTSFRandomMultiStack(random_state=random_state, n_jobs=n_jobs, selection="best-base")
    elif model_name == "multistack-best":
        return LokyStackerV10RSTSFRandomMultiStack(random_state=random_state, n_jobs=n_jobs, selection="best")
    elif model_name == "multistack-ridgecv":
        return LokyStackerV10RSTSFRandomMultiStack(random_state=random_state, n_jobs=n_jobs, selection=None)
    elif model_name == "mycatch22v2":
        return Catch22Classifier(random_state=random_state + 1000)
    elif model_name == "mymrhydra":
        return MultiRocketHydraClassifier(random_state=random_state, n_jobs=n_jobs)
    elif model_name == "mymrhydrav2":
        return MultiRocketHydraClassifier(random_state=random_state + 1000, n_jobs=n_jobs)
    elif model_name == "loky-stacker-v10-tabicl":
        return LokyStackerV10TabICL(random_state=random_state, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v10-fm":
        return LokyStackerV10FM(random_state=random_state, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v10-fm-tsfresh":
        return LokyStackerV10FMTSFresh(random_state=random_state, n_jobs=n_jobs, verbose=10)
    elif model_name == "loky-stacker-v10-rstsf-random":
        return LokyStackerV10RSTSFRandom(random_state=random_state, n_jobs=n_jobs, verbose=10)
    elif model_name == "ablation-multirockethydra-bestk-p-ridgecv":
        return make_ablation_model("multirockethydra-bestk-p-ridgecv", random_state, n_jobs, verbose=10)
    elif model_name == "ablation-quant-etc":
        return make_ablation_model("quant-etc", random_state, n_jobs, verbose=10)
    elif model_name == "ablation-rdst-p-ridgecv":
        return make_ablation_model("rdst-p-ridgecv", random_state, n_jobs, verbose=10)
    elif model_name == "ablation-rstsf-random-etc":
        return make_ablation_model("rstsf-random-etc", random_state, n_jobs, verbose=10)
    elif model_name == "ablation-fm-p-ridgecv":
        return make_ablation_model("fm-p-ridgecv", random_state, n_jobs, verbose=10)
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
    elif model_name == "mantis+chronos2+diff-ridgecv":
        return make_tsfm_model("mantis+chronos2-ridgecv", random_state=random_state, use_diff=True)
    elif model_name == "tabicl":
        return TabICLTimeSeriesClassifier(random_state=random_state, device="cuda")
    elif model_name == "tabicl-diff":
        return TabICLTimeSeriesClassifier(random_state=random_state, device="cuda", include_diff=True)
    elif model_name == "rstsf-random":
        return RSTSFRandom(n_estimators=200, n_intervals=600, random_state=random_state, n_jobs=n_jobs)
    elif model_name == "rstsf-random-ridge":
        return RSTSFRandom(n_estimators=200, n_intervals=600, estimator=RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10)), random_state=random_state, n_jobs=n_jobs)
    elif model_name == "rstsf-unsupervised":
        return RSTSFUnsupervised(n_estimators=200, n_intervals=50, random_state=random_state, n_jobs=n_jobs)
    elif model_name == "rstsf-unsupervised-ridge":
        return RSTSFUnsupervised(n_estimators=200, n_intervals=50, estimator=RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10)), random_state=random_state, n_jobs=n_jobs)
    elif model_name == "rstsf-combined":
        return RSTSFCombined(n_estimators=200, n_intervals_random=600, n_intervals_unsupervised=50, random_state=random_state, n_jobs=n_jobs)
    elif model_name == "rstsf-combined-ridge":
        return RSTSFCombined(n_estimators=200, n_intervals_random=600, n_intervals_unsupervised=50, estimator=RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10)), random_state=random_state, n_jobs=n_jobs)
    elif model_name == "rstsf-unsupervised-raw":
        return RSTSFUnsupervisedRaw(n_intervals=50, random_state=random_state, n_jobs=n_jobs)
    elif model_name == "hivecotev2-4h-j8":
        return HIVECOTEV2(time_limit_in_minutes=240, n_jobs=n_jobs, random_state=random_state)
    elif model_name == "hivecotev2-1h-j8":
        return HIVECOTEV2(time_limit_in_minutes=60, n_jobs=n_jobs, random_state=random_state)
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
    "mr-hydra-kbest-5000",
    "mr-hydra-kbest-10000",
    "mr-hydra-kbest-30000",
    "mr-hydra-kbest-auto",
    "mr-hydra-contained-auto",
    "loky-stacker-v10-base",
    "loky-stacker-v10-tabicl",
    "loky-stacker-v10-fm",
    "loky-stacker-v10-fm-tsfresh",
    "loky-stacker-v10-rstsf-random",
    "loky-stacker-v10-base-2x",
    "loky-stacker-v10-base-5x",
    "loky-stacker-v10-base-r3",
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
    "mantis+chronos2+diff-ridgecv",
    "tabicl",
    "tabicl-diff",
    "mydummy",
    "mycatch22",
    "TSCGlue-3-3-26",
    "TSCGlue-17-4-26",
    "multistack-best-stacking",
    "multistack-best-base",
    "multistack-best",
    "multistack-ridgecv",
    "mycatch22v2",
    "mymrhydra",
    "mymrhydrav2",
    "rstsf-random",
    "rstsf-random-ridge",
    "rstsf-unsupervised",
    "rstsf-unsupervised-ridge",
    "rstsf-combined",
    "rstsf-combined-ridge",
    "rstsf-unsupervised-raw",
    #*_FILTER_VARIANTS,
    "ablation-multirockethydra-bestk-p-ridgecv",
    "ablation-quant-etc",
    "ablation-rdst-p-ridgecv",
    "ablation-rstsf-random-etc",
    "ablation-fm-p-ridgecv",
    "hivecotev2-4h-j8",
    "hivecotev2-1h-j8",
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
