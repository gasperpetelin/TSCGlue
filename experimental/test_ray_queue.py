"""
Example of Ray queue pattern for distributed model training.
This reduces overhead by having n workers process k models (k >> n).
With streaming results: collect trained models while workers are still running.
"""

import time

import numpy as np
import ray
from ray.util.queue import Queue
from sklearn.ensemble import RandomForestRegressor


@ray.remote
def train_worker(model_queue: Queue, result_queue: Queue, X_train, y_train, worker_id):
    """Worker that processes multiple models from queue and streams results"""
    processed_count = 0

    print(f"Worker {worker_id} started")

    while True:
        try:
            # Non-blocking get with timeout
            model_info = model_queue.get(timeout=1)

            if model_info is None:  # Poison pill to stop
                print(f"Worker {worker_id} received stop signal")
                break

            model_name, model = model_info
            print(f"Worker {worker_id} processing {model_name}")

            # Train the model
            trained_model = model.fit(X_train, y_train)

            # Stream result immediately to result queue
            result_queue.put((model_name, trained_model))
            processed_count += 1

        except Exception as e:
            print(f"Worker {worker_id} queue empty or error: {e}")
            break

    print(f"Worker {worker_id} finished. Processed {processed_count} models")
    # Signal this worker is done
    result_queue.put(("__WORKER_DONE__", worker_id))


def approach_queue_streaming(models_dict, X_train, y_train, n_workers=3):
    """Using Ray Queue with streaming results"""
    print("\n" + "=" * 60)
    print("Ray Queue Pattern with Streaming Results")
    print("=" * 60)

    start_time = time.time()

    # Put data in Ray object store once
    X_ref = ray.put(X_train)
    y_ref = ray.put(y_train)

    # Create queues
    model_queue = Queue()
    result_queue = Queue()

    # Populate model queue
    k = len(models_dict)
    for name, model in models_dict.items():
        model_queue.put((name, model))

    # Add poison pills for workers to know when to stop
    for _ in range(n_workers):
        model_queue.put(None)

    # Start n workers (non-blocking)
    futures = [
        train_worker.remote(model_queue, result_queue, X_ref, y_ref, worker_id=i)
        for i in range(n_workers)
    ]

    # Collect results as they arrive
    trained_models = {}
    workers_done = 0

    print(f"\nCollecting results as they arrive (expecting {k} models from {n_workers} workers)...")

    while workers_done < n_workers:
        try:
            result = result_queue.get(timeout=1)
            model_name, model_or_id = result

            if model_name == "__WORKER_DONE__":
                workers_done += 1
                print(f"Worker {model_or_id} completed. ({workers_done}/{n_workers} workers done)")
            else:
                trained_models[model_name] = model_or_id
                print(f"Received {model_name}. Total collected: {len(trained_models)}/{k}")

        except Exception:
            # Timeout or queue empty, check if workers are still running
            pass

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Trained {len(trained_models)} models using {n_workers} workers")

    return trained_models


@ray.remote
def train_single_model(model_name, model, X_train, y_train):
    """Traditional approach: one task per model"""
    print(f"Task for {model_name} started")
    trained = model.fit(X_train, y_train)
    print(f"Task for {model_name} completed")
    return model_name, trained


def approach_traditional(models_dict, X_train, y_train):
    """Traditional one-task-per-model (for comparison)"""
    print("\n" + "=" * 60)
    print("Traditional (one task per model)")
    print("=" * 60)

    start_time = time.time()

    # Put data in Ray object store once
    X_ref = ray.put(X_train)
    y_ref = ray.put(y_train)

    # Create one task per model
    futures = [
        train_single_model.remote(name, model, X_ref, y_ref) for name, model in models_dict.items()
    ]

    # Collect results
    results = ray.get(futures)
    trained_models = dict(results)

    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s")
    print(f"Trained {len(trained_models)} models using {len(futures)} tasks")

    return trained_models


def main():
    # Initialize Ray with 6 workers
    ray.init(ignore_reinit_error=True, num_cpus=6)

    # Create dataset
    np.random.seed(42)
    X_train = np.random.randn(1000, 20)
    y_train = np.random.randn(1000)

    # Create 100 random forests with n_jobs=1
    models_dict = {
        f"rf_{i}": RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=i)
        for i in range(100)
    }

    k = len(models_dict)
    print(f"Training {k} Random Forest models...")
    print(f"Ray cluster has {ray.cluster_resources()['CPU']} CPUs available")

    # Test Queue pattern with streaming (n=6 workers)
    n_workers = 20
    trained_1 = approach_queue_streaming(models_dict.copy(), X_train, y_train, n_workers=n_workers)

    # Test Traditional (for comparison)
    # Recreate models since they were trained in previous approach
    models_dict_copy = {
        f"rf_{i}": RandomForestRegressor(n_estimators=100, n_jobs=1, random_state=i)
        for i in range(100)
    }
    trained_2 = approach_traditional(models_dict_copy, X_train, y_train)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Queue pattern with streaming results:")
    print(f"- Fewer Ray tasks created ({n_workers} vs {k})")
    print("- Data passed to object store only once")
    print("- Workers process multiple models sequentially")
    print("- Results available immediately as each model finishes")

    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    main()
