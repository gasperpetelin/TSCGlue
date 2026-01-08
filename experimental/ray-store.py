import gc
import sys
import time

import numpy as np
import ray


def main():
    ray.init(ignore_reinit_error=True)

    print(f"Ray version: {ray.__version__}\n")

    # Create objects
    data1 = np.random.rand(10000, 10000)
    data2 = [i for i in range(1000000)]
    data3 = {f"key_{i}": f"value_{i}" for i in range(100000)}
    data4 = "Hello World! " * 100000
    data5 = np.zeros((20000, 20000))

    print("Approximate object sizes:")
    print("-" * 50)
    print(f"Numpy array (10000x10000): {sys.getsizeof(data1) / (1024**2):.2f} MB")
    print(f"List (1M elements): {sys.getsizeof(data2) / (1024**2):.2f} MB")
    print(f"Dictionary (100k items): {sys.getsizeof(data3) / (1024**2):.2f} MB")
    print(f"String (repeated): {sys.getsizeof(data4) / (1024**2):.2f} MB")
    print(f"Numpy array (20000x20000): {sys.getsizeof(data5) / (1024**2):.2f} MB")

    print("\n" + "=" * 50)
    print("Object store memory BEFORE putting objects:")
    print("=" * 50)

    total_resources = ray.cluster_resources()
    object_store_total = total_resources.get("object_store_memory", 0)

    available_resources = ray.available_resources()
    object_store_available = available_resources.get("object_store_memory", 0)
    object_store_used = object_store_total - object_store_available

    print(f"Total: {object_store_total / (1024**3):.2f} GB")
    print(f"Used: {object_store_used / (1024**3):.2f} GB ({object_store_used / (1024**2):.2f} MB)")
    print(f"Available: {object_store_available / (1024**3):.2f} GB")
    print(f"Usage: {(object_store_used / object_store_total) * 100:.1f}%")

    print("\n" + "=" * 50)
    print("Putting objects in Ray object store...")
    print("=" * 50 + "\n")

    obj1 = ray.put(data5)
    obj2 = ray.put(data2)
    obj3 = ray.put(data3)
    obj4 = ray.put(data4)
    obj5 = ray.put(data1)
    obj6 = ray.put(1)
    obj7 = ray.put("small object")

    print("Objects stored successfully!\n")

    time.sleep(2)

    available_resources = ray.available_resources()
    object_store_available = available_resources.get("object_store_memory", 0)
    object_store_used = object_store_total - object_store_available

    print("Object store memory AFTER putting objects:")
    print("-" * 50)
    print(f"Total: {object_store_total / (1024**3):.2f} GB")
    print(f"Used: {object_store_used / (1024**3):.2f} GB ({object_store_used / (1024**2):.2f} MB)")
    print(f"Available: {object_store_available / (1024**3):.2f} GB")
    print(f"Usage: {(object_store_used / object_store_total) * 100:.1f}%")

    print("\n" + "=" * 50)
    print("Deleting objects...")
    print("=" * 50 + "\n")

    del obj1, obj2, obj3, obj4, obj5, obj6, obj7

    time.sleep(5)

    # Force Python GC
    gc.collect()
    from ray._private.internal_api import global_gc

    global_gc()
    from ray._private.internal_api import memory_summary

    summary = memory_summary(stats_only=True)
    print("Summary --------------------------------------------")
    print(summary)
    print("Summary --------------------------------------------")

    # Wait for Ray to reclaim memory
    time.sleep(60)

    print("\nObject store memory AFTER deletion:")
    print("-" * 50)

    summary = memory_summary(stats_only=True)
    print("Summary --------------------------------------------")
    print(summary)
    print("Summary --------------------------------------------")

    available_resources = ray.available_resources()
    object_store_available = available_resources.get("object_store_memory", 0)
    object_store_used = object_store_total - object_store_available

    print(f"Total: {object_store_total / (1024**3):.2f} GB")
    print(f"Used: {object_store_used / (1024**3):.2f} GB ({object_store_used / (1024**2):.2f} MB)")
    print(f"Available: {object_store_available / (1024**3):.2f} GB")
    print(f"Usage: {(object_store_used / object_store_total) * 100:.1f}%")

    ray.shutdown()


if __name__ == "__main__":
    main()
