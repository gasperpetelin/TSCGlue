import os

from autotsc.models2 import RSTSFUnsupervisedClassifier

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
from urllib.parse import urlparse

import boto3
import click
import polars as pl
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.interval_based import RSTSF, QUANTClassifier, DrCIFClassifier
from aeon.classification.shapelet_based import RDSTClassifier
from aeon.pipeline import make_pipeline as aeon_make_pipeline
from aeon.transformations.collection import Normalizer
from aeon.datasets.tsc_datasets import univariate
from aeon.benchmarking.resampling import stratified_resample_data
from botocore.exceptions import ClientError
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

from autotsc import transformers, utils
from autotsc.models import StackerV4, FastStackerV4, FastStackerV5, LokyStackerV5


def s3_file_exists(s3_uri: str) -> bool:
    s3 = boto3.client("s3")
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise  # propagate other errors (e.g., permission, throttling)


def get_model(model_name, random_state):
    if model_name == "mr-hydra":
        return MultiRocketHydraClassifier(n_jobs=16, random_state=random_state)
    elif model_name == "quant":
        return QUANTClassifier(random_state=random_state)
    elif model_name == "quant-catboost":
        cat = CatBoostClassifier(
            random_seed=random_state,
            verbose=True,
            thread_count=16,
        )
        return QUANTClassifier(estimator=cat, random_state=random_state)
    elif model_name == "rdst":
        return RDSTClassifier(n_jobs=16, random_state=random_state)
    elif model_name == "rstsf":
        return RSTSF(random_state=random_state, n_jobs=16)
    elif model_name == "hivecotev2":
        return HIVECOTEV2(random_state=random_state, n_jobs=16, verbose=10)
    elif model_name == "stacker-v4-r3":
        return StackerV4(random_state=random_state, n_repetitions=3)
    elif model_name == "stacker-v4-r1":
        return StackerV4(random_state=random_state, n_repetitions=1)
    elif model_name == "fast-stacker-v4-r1":
        return FastStackerV4(random_state=random_state, n_repetitions=1, n_jobs=16)
    elif model_name == "fast-stacker-v5-r1":
        return FastStackerV5(random_state=random_state, n_repetitions=1, n_jobs=16)
    elif model_name == "fast-stacker-v5-r3":
        return FastStackerV5(random_state=random_state, n_repetitions=3, n_jobs=16)
    elif model_name == "loky-stacker-v5-r1":
        return LokyStackerV5(random_state=random_state, n_repetitions=1, n_jobs=8)
    elif model_name == "catch22":
        return Catch22Classifier(n_jobs=16)
    elif model_name == "drcif":
        return DrCIFClassifier(n_jobs=16, random_state=random_state, n_estimators=20)
    elif model_name == "u-rstsf":
        return RSTSFUnsupervisedClassifier(n_jobs=16, random_state=random_state)
    elif model_name == "cumsum-mr-hydra":
        return aeon_make_pipeline(
            transformers.CumSum(), MultiRocketHydraClassifier(n_jobs=16, random_state=random_state)
        )
    elif model_name == "scale-mr-hydra":
        return aeon_make_pipeline(
            Normalizer(), MultiRocketHydraClassifier(n_jobs=16, random_state=random_state)
        )
    elif model_name == "polar-angle-mr-hydra":
        return aeon_make_pipeline(
            transformers.PolarCoordinates(mode="angle"),
            MultiRocketHydraClassifier(n_jobs=16, random_state=random_state),
        )
    elif model_name == "polar-magnitude-mr-hydra":
        return aeon_make_pipeline(
            transformers.PolarCoordinates(mode="magnitude"),
            MultiRocketHydraClassifier(n_jobs=16, random_state=random_state),
        )
    elif model_name == "rank-mr-hydra":
        return aeon_make_pipeline(
            transformers.RankTransform(),
            MultiRocketHydraClassifier(n_jobs=16, random_state=random_state),
        )
    elif model_name == "difference-mr-hydra":
        return aeon_make_pipeline(
            transformers.Difference(),
            MultiRocketHydraClassifier(n_jobs=16, random_state=random_state),
        )
    elif model_name == "downsample-mr-hydra":
        return aeon_make_pipeline(
            transformers.DownsampleTransformer(proportion=0.5),
            transformers.PadToLengthTransformer(target_length=10),
            MultiRocketHydraClassifier(n_jobs=16, random_state=random_state),
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


ALL_MODELS = [
    "rstsf", "mr-hydra", "quant", "rdst", "catch22", "drcif", "u-rstsf",
    "stacker-v4-r1", "loky-stacker-v5-r1", "hivecotev2",
    "cumsum-mr-hydra", "scale-mr-hydra", "polar-angle-mr-hydra", "polar-magnitude-mr-hydra",
    "rank-mr-hydra", "difference-mr-hydra", "downsample-mr-hydra", "quant-catboost"
]


@click.command()
@click.option("-m", "--models", multiple=True, help="Models to run (can be specified multiple times or comma-separated)")
@click.option("-d", "--datasets", "dataset_names", multiple=True, help="Datasets to run (can be specified multiple times or comma-separated)")
@click.option("-r", "--resample", type=int, default=None, help="Resample data: 1=yes, 0=no, unset=both")
@click.option("-l", "--list-models", is_flag=True, help="List all available models and exit")
@click.option("--list-datasets", is_flag=True, help="List all available datasets and exit")
def main(models, dataset_names, resample, list_models, list_datasets):
    """Run stacking experiments on TSC datasets."""
    import random
    from itertools import product

    all_datasets = list(univariate)

    # Handle --list-models flag
    if list_models:
        click.echo("Available models:")
        for model in ALL_MODELS:
            click.echo(f"  - {model}")
        return

    # Handle --list-datasets flag
    if list_datasets:
        click.echo("Available datasets:")
        for ds in all_datasets:
            click.echo(f"  - {ds}")
        return

    # Determine which models to run
    if models:
        # Support both multiple -m flags and comma-separated values
        model_list = []
        for m in models:
            model_list.extend([x.strip() for x in m.split(",")])
        # Validate specified models
        invalid_models = [m for m in model_list if m not in ALL_MODELS]
        if invalid_models:
            click.echo(f"Error: Unknown models: {', '.join(invalid_models)}", err=True)
            click.echo("Use -l to list available models", err=True)
            raise click.Abort()
        model_names = model_list
        click.echo(f"Running models: {', '.join(model_names)}")
    else:
        model_names = ALL_MODELS
        click.echo(f"Running all models")

    # Determine which datasets to run
    if dataset_names:
        # Support both multiple -d flags and comma-separated values
        dataset_list = []
        for d in dataset_names:
            dataset_list.extend([x.strip() for x in d.split(",")])
        # Validate specified datasets
        invalid_datasets = [d for d in dataset_list if d not in all_datasets]
        if invalid_datasets:
            click.echo(f"Error: Unknown datasets: {', '.join(invalid_datasets)}", err=True)
            click.echo("Use --list-datasets to list available datasets", err=True)
            raise click.Abort()
        datasets = dataset_list
        click.echo(f"Running datasets: {', '.join(datasets)}")
    else:
        datasets = all_datasets
        click.echo(f"Running all datasets")

    # Determine resample options
    if resample is None:
        resample_options = [False, True]
        click.echo("Running both resampled and non-resampled")
    else:
        resample_options = [bool(resample)]
        click.echo(f"Running {'resampled' if resample else 'non-resampled'} only")

    write_dir = "s3://tsc-glue/performance"
    runs = [100, 200, 300, 400, 500]

    combos = list(product(datasets, model_names, runs, resample_options))
    random.shuffle(combos)

    n = len(combos)
    for k, (dataset, model_name, run, resampled) in enumerate(combos, 1):
        try:
            stats = {
                "dataset": dataset,
                "model": model_name,
                "run": run,
                "resampled": resampled,
            }

            hash_val = pl.DataFrame([stats]).hash_rows(seed=42, seed_1=1, seed_2=2, seed_3=3).item()
            file = f"{write_dir}/{hash_val}.parquet"

            # Check if file exists in S3
            if s3_file_exists(file):
                print(f"[{k}/{n}] Skipping: Dataset={dataset}, Run={run}, Model={model_name}, Resampled={resampled}")
                continue
            else:
                print(f"[{k}/{n}] Processing: Dataset={dataset}, Run={run}, Model={model_name}, Resampled={resampled}")

            X_train, y_train, X_test, y_test = utils.load_dataset(dataset)

            if resampled:
                X_train, y_train, X_test, y_test = stratified_resample_data(
                    X_train, y_train, X_test, y_test, random_state=run
                )

            print(f"Running {dataset} with model {model_name} run {run} resampled={resampled}")

            model = get_model(model_name, random_state=run)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            stats["test_accuracy"] = acc

            df_stat = pl.DataFrame([stats])
            df_stat.write_parquet(file)
        except Exception as e:
            print(f"Error processing Dataset={dataset}, Run={run}, Model={model_name}, Resampled={resampled}: {e}")


if __name__ == "__main__":
    main()
