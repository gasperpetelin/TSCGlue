import random
import time
from pathlib import Path

import click
import polars as pl
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.hybrid import HIVECOTEV2
from sklearn.metrics import accuracy_score

from run_stacking2 import ALL_MODELS, discover_datasets, discover_folds, get_model
from tscglue.data_loader import load_fold
from tscglue.utils import LocalFileCache, S3FileCache


ROOT_DIR = Path(__file__).resolve().parent.parent
DISK_RESULTS_DIR = ROOT_DIR / "results" / "timing_runs_slurm"
S3_RESULTS_DIR = "s3://tsc-glue/performance-timing-slurm"
TIMING_MODEL_NAMES = [
    "multirockethydra",
    "hcv2",
    "TSCGlue-17-4-26",
]


def parse_csv_args(values: tuple[str, ...]) -> list[str]:
    parsed: list[str] = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed


def available_model_names() -> list[str]:
    return sorted(set(ALL_MODELS) | set(TIMING_MODEL_NAMES))


def build_model(model_name: str, random_state: int, n_train: int, n_jobs: int):
    if model_name == "multirockethydra":
        return MultiRocketHydraClassifier(random_state=random_state, n_jobs=n_jobs)
    if model_name == "hcv2":
        return HIVECOTEV2(time_limit_in_minutes=240, n_jobs=n_jobs, random_state=random_state)
    return get_model(model_name, random_state=random_state, n_train=n_train, n_jobs=n_jobs)


def make_key(dataset: str, model_name: str, fold: int, n_jobs: int) -> dict[str, int | str]:
    return {
        "dataset": dataset,
        "model": model_name,
        "fold": fold,
        "n_jobs": n_jobs,
    }


def make_filename(key: dict[str, int | str]) -> str:
    hash_value = (
        pl.DataFrame([key])
        .hash_rows(seed=42, seed_1=1, seed_2=2, seed_3=3)
        .item()
    )
    return f"{hash_value}.parquet"


def get_cache(storage: str):
    if storage == "s3":
        return S3FileCache(S3_RESULTS_DIR)
    return LocalFileCache(str(DISK_RESULTS_DIR))


def read_results(cache, filenames: list[str]) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for filename in filenames:
        if cache.exists(filename):
            frames.append(cache.read_parquet(filename))
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="vertical")


def print_summary(df: pl.DataFrame) -> None:
    if df.is_empty():
        click.echo("No successful timing rows found for the selected combinations.")
        return

    summary = (
        df.group_by(["model", "n_jobs"])
        .agg(
            [
                pl.len().alias("n_runs"),
                pl.col("fit_time_s").sum().alias("fit_sum_s"),
                pl.col("predict_time_s").sum().alias("predict_sum_s"),
                pl.col("total_time_s").sum().alias("total_sum_s"),
                pl.col("total_time_s").mean().alias("total_mean_s"),
                (pl.col("fit_time_s") / pl.col("n_train")).mean().alias("mean_train_s_per_inst"),
                (pl.col("predict_time_s") / pl.col("n_test")).mean().alias("mean_test_s_per_inst"),
                pl.col("accuracy").mean().alias("accuracy_mean"),
            ]
        )
        .sort(["model", "n_jobs"])
    )

    click.echo("")
    click.echo("Summary:")
    for row in summary.iter_rows(named=True):
        train_inst_per_s = (
            1.0 / row["mean_train_s_per_inst"]
            if row["mean_train_s_per_inst"] and row["mean_train_s_per_inst"] > 0
            else float("nan")
        )
        test_inst_per_s = (
            1.0 / row["mean_test_s_per_inst"]
            if row["mean_test_s_per_inst"] and row["mean_test_s_per_inst"] > 0
            else float("nan")
        )
        click.echo(
            "  "
            f"{row['model']} (j={row['n_jobs']}): "
            f"runs={row['n_runs']}, "
            f"total={row['total_sum_s'] / 60:.2f} min, "
            f"fit={row['fit_sum_s'] / 60:.2f} min, "
            f"predict={row['predict_sum_s'] / 60:.2f} min, "
            f"mean_total/dataset={row['total_mean_s'] / 60:.2f} min, "
            f"train_inst/s={train_inst_per_s:.2f}, "
            f"test_inst/s={test_inst_per_s:.2f}, "
            f"mean_acc={row['accuracy_mean']:.4f}"
        )


@click.command()
@click.option(
    "-m",
    "--models",
    multiple=True,
    help="Models to run (can be specified multiple times or comma-separated).",
)
@click.option(
    "-d",
    "--datasets",
    "dataset_names",
    multiple=True,
    help="Datasets to run (can be specified multiple times or comma-separated).",
)
@click.option(
    "-f",
    "--folds",
    "fold_spec",
    default="5",
    show_default=True,
    help="Folds to run (comma-separated, e.g. '5' or '0,1,2').",
)
@click.option("-l", "--list-models", is_flag=True, help="List all available models and exit.")
@click.option("--list-datasets", is_flag=True, help="List all available datasets and exit.")
@click.option(
    "--storage",
    type=click.Choice(["s3", "disk"]),
    default="disk",
    show_default=True,
    help="Storage backend for parquet timing rows.",
)
@click.option("-j", "--n-jobs", default=16, type=int, show_default=True, help="Number of jobs passed to the model constructor.")
@click.option("--overwrite", is_flag=True, help="Re-run combinations even if a cached parquet already exists.")
@click.option("--summary-only", is_flag=True, help="Do not run models; only print a summary for the selected combinations already on disk/in S3.")
def main(
    models,
    dataset_names,
    fold_spec,
    list_models,
    list_datasets,
    storage,
    n_jobs,
    overwrite,
    summary_only,
):
    """Run timing benchmarks for fit/predict on local fold datasets."""
    all_datasets = discover_datasets()

    if list_models:
        click.echo("Available models:")
        for model_name in available_model_names():
            click.echo(f"  - {model_name}")
        return

    if list_datasets:
        click.echo("Available datasets:")
        for dataset in all_datasets:
            folds = discover_folds(dataset)
            click.echo(f"  - {dataset} ({len(folds)} folds)")
        return

    if models:
        model_names = parse_csv_args(models)
        valid_model_names = set(available_model_names())
        invalid_models = [model_name for model_name in model_names if model_name not in valid_model_names]
        if invalid_models:
            click.echo(f"Error: unknown models: {', '.join(invalid_models)}", err=True)
            click.echo("Use -l to list available models.", err=True)
            raise click.Abort()
    else:
        model_names = list(TIMING_MODEL_NAMES)

    if dataset_names:
        datasets = parse_csv_args(dataset_names)
        invalid_datasets = [dataset for dataset in datasets if dataset not in all_datasets]
        if invalid_datasets:
            click.echo(f"Error: unknown datasets: {', '.join(invalid_datasets)}", err=True)
            click.echo("Use --list-datasets to list available datasets.", err=True)
            raise click.Abort()
    else:
        datasets = all_datasets

    requested_folds = [int(part.strip()) for part in fold_spec.split(",") if part.strip()]
    cache = get_cache(storage)

    combos: list[tuple[str, str, int]] = []
    skipped_missing_folds = 0
    for dataset in datasets:
        available_folds = set(discover_folds(dataset))
        for fold in requested_folds:
            if fold not in available_folds:
                click.echo(f"Skipping dataset={dataset}: fold {fold} is not available.")
                skipped_missing_folds += 1
                continue
            for model_name in model_names:
                combos.append((dataset, model_name, fold))

    random.shuffle(combos)
    click.echo(
        f"Running {len(combos)} timing combinations "
        f"(models={len(model_names)}, datasets={len(datasets)}, folds={requested_folds}, n_jobs={n_jobs})."
    )
    if skipped_missing_folds:
        click.echo(f"Skipped {skipped_missing_folds} dataset/fold pairs with no matching local fold files.")

    filenames = [make_filename(make_key(dataset, model_name, fold, n_jobs)) for dataset, model_name, fold in combos]

    if summary_only:
        print_summary(read_results(cache, filenames))
        return

    for index, (dataset, model_name, fold) in enumerate(combos, 1):
        key = make_key(dataset, model_name, fold, n_jobs)
        filename = make_filename(key)
        if cache.exists(filename) and not overwrite:
            click.echo(f"[{index}/{len(combos)}] SKIP dataset={dataset} fold={fold} model={model_name} j={n_jobs}")
            continue

        click.echo(f"[{index}/{len(combos)}] RUN  dataset={dataset} fold={fold} model={model_name} j={n_jobs}")

        model = None
        try:
            X_train, y_train, X_test, y_test = load_fold(dataset, fold)
            model = build_model(model_name, random_state=fold, n_train=len(X_train), n_jobs=n_jobs)

            fit_start = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time_s = time.perf_counter() - fit_start

            predict_start = time.perf_counter()
            y_pred = model.predict(X_test)
            predict_time_s = time.perf_counter() - predict_start

            accuracy = accuracy_score(y_test, y_pred)
            row = {
                **key,
                "n_train": int(X_train.shape[0]),
                "n_test": int(X_test.shape[0]),
                "n_channels": int(X_train.shape[1]),
                "series_len": int(X_train.shape[2]),
                "n_classes": int(len(set(y_train))),
                "fit_time_s": float(fit_time_s),
                "predict_time_s": float(predict_time_s),
                "total_time_s": float(fit_time_s + predict_time_s),
                "accuracy": float(accuracy),
            }

            cache.add(pl.DataFrame([row]), filename)
            click.echo(
                "           "
                f"fit={fit_time_s:.2f}s pred={predict_time_s:.2f}s "
                f"total={fit_time_s + predict_time_s:.2f}s acc={accuracy:.4f}"
            )
        except Exception as exc:
            click.echo(
                f"[{index}/{len(combos)}] ERROR dataset={dataset} fold={fold} model={model_name} j={n_jobs}: {exc}",
                err=True,
            )
        finally:
            if model is not None and hasattr(model, "cleanup"):
                model.cleanup()

    print_summary(read_results(cache, filenames))


if __name__ == "__main__":
    main()
