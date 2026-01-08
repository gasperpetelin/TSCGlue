import os

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_DISABLE_METRICS"] = "1"
os.environ["RAY_DISABLE_METRICS_EXPORT"] = "1"

os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["RAY_memory_usage_threshold"] = "0.999"

import random
import subprocess
from time import perf_counter

import polars as pl
from aeon.classification.convolution_based import MultiRocketClassifier
from aeon.classification.hybrid import HIVECOTEV2
from aeon.datasets.tsc_datasets import univariate
from sklearn.metrics import accuracy_score
from sklearn.linear_model import RidgeClassifierCV

from autotsc import utils
from autotsc.models import AutoTSCModel
import ray

def create_model(model_name):
    if model_name == "autotsc":
        model = AutoTSCModel(n_jobs=-1, verbose=0)
        return model, True
    elif model_name == "hivecotev2":
        model = HIVECOTEV2(n_jobs=-1, time_limit_in_minutes=3, verbose=False)
        return model, False
    elif model_name == "multirocket":
        model = MultiRocketClassifier(n_jobs=-1)
        return model, False
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def run_benchmark(n_runs=1):
    datasets = list(univariate)
    random.shuffle(datasets)
    model_names = ["autotsc", "hivecotev2", "multirocket"]

    os.makedirs("benchmark_results", exist_ok=True)
    # all_results = []

    for i, dataset_name in enumerate(datasets, 1):
        try:
            X_train, y_train, X_test, y_test = utils.load_dataset(dataset_name)

            dataset_info = {
                "dataset": dataset_name,
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
                "series_length": X_train.shape[2],
                "n_classes": len(set(y_train)),
            }

            for model_name in model_names:
                for run in range(n_runs):
                    filename = f"{dataset_name}_{model_name}_run{run}.parquet"
                    filepath = f"benchmark_results/{filename}"

                    if os.path.exists(filepath):
                        print(f"[{i}/{len(datasets)}] {dataset_name} | {model_name} | run {run} | SKIPPED")
                        continue

                    print(f"[{i}/{len(datasets)}] {dataset_name} | {model_name} | run {run}")

                    try:
                        model, needs_ray = create_model(model_name)

                        if needs_ray:
                            with utils.ray_init_or_reuse(num_cpus=24, resources={"meta": 100}, ignore_reinit_error=True):
                                start_time = perf_counter()
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                                elapsed_time = perf_counter() - start_time
                        else:
                            start_time = perf_counter()
                            model.fit(X_train, y_train)
                            predictions = model.predict(X_test)
                            elapsed_time = perf_counter() - start_time

                        accuracy = accuracy_score(y_test, predictions)

                        result = {
                            **dataset_info,
                            "model": model_name,
                            "run": run,
                            "accuracy": accuracy,
                            "time": elapsed_time,
                        }

                        pl.DataFrame([result]).write_parquet(filepath)
                        # all_results.append(result)

                    except Exception as e:
                        print('error', e)
                    ray.shutdown()
                    # subprocess.run(["uv", "cache", "clean"], check=False)

        except Exception as e:
            continue

    #pl.DataFrame(all_results).write_parquet("benchmark_results/all_results_summary.parquet")
    #print(f"\nDone! Results in benchmark_results/")

    #return pl.DataFrame(all_results)


if __name__ == "__main__":
    results = run_benchmark(n_runs=1)
