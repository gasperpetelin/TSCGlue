"""Utility functions for AutoTSC."""

import os
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import boto3
import polars as pl
from aeon.datasets import load_classification
from botocore.exceptions import ClientError
from sklearn.model_selection import KFold, StratifiedKFold


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
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
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

    def list_files(self) -> list[str]:
        self._load_once()
        return list(self._files)

    def read_parquet(self, filename: str) -> pl.DataFrame:
        return pl.read_parquet(f"{self.base_s3_dir}/{filename}")

    def read_all_parquet(self) -> pl.DataFrame:
        return pl.read_parquet(f"{self.base_s3_dir}/*.parquet")

    def read_all_parquet_cached(
        self, cache_dir: str | None = None, max_workers: int = 16
    ) -> pl.DataFrame:
        from tqdm import tqdm

        subdir = f"{self.bucket}__{self.prefix.replace('/', '_')}"
        local_dir = Path(cache_dir) if cache_dir else Path(tempfile.gettempdir()) / "tsc-glue-cache" / subdir
        local_dir.mkdir(parents=True, exist_ok=True)

        local_files = {f.name for f in local_dir.iterdir() if f.suffix == ".parquet"}

        self._load_once()
        remote_new = [f for f in self._files if f.endswith(".parquet") and f not in local_files]

        def _download(filename):
            self._s3.download_file(self.bucket, self._full_key(filename), str(local_dir / filename))

        if remote_new:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_download, f) for f in remote_new]
                for _ in tqdm(as_completed(futures), total=len(futures)):
                    pass

        paths = sorted(local_dir.glob("*.parquet"))
        return pl.read_parquet(paths)


class LocalFileCache:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def exists(self, filename: str) -> bool:
        return (self.base_dir / filename).exists()

    def add(self, df: pl.DataFrame, filename: str):
        df.write_parquet(self.base_dir / filename)

    def list_files(self) -> list[str]:
        return [f.name for f in self.base_dir.iterdir() if f.suffix == ".parquet"]

    def read_parquet(self, filename: str) -> pl.DataFrame:
        return pl.read_parquet(self.base_dir / filename)


def load_s3_parquet_cached(
    s3_prefix: str = "s3://tsc-glue/performance-benchmarking/",
    max_workers: int = 16,
) -> pl.DataFrame:
    """Download all parquet files from an S3 prefix, cache locally, and return as a DataFrame."""
    import boto3
    from tqdm import tqdm

    cache_dir = os.path.join(tempfile.gettempdir(), "tsc-glue-cache")
    os.makedirs(cache_dir, exist_ok=True)

    parsed = urlparse(s3_prefix)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    s3 = boto3.client("s3")

    local_files = {f for f in os.listdir(cache_dir) if f.endswith(".parquet")}

    paginator = s3.get_paginator("list_objects_v2")
    remote_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".parquet"):
                fname = key.rsplit("/", 1)[-1]
                if fname not in local_files:
                    remote_keys.append(key)

    def _download(key):
        fname = key.rsplit("/", 1)[-1]
        s3.download_file(bucket, key, os.path.join(cache_dir, fname))

    if remote_keys:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_download, k) for k in remote_keys]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass

    local_paths = sorted(
        os.path.join(cache_dir, f)
        for f in os.listdir(cache_dir)
        if f.endswith(".parquet")
    )
    return pl.read_parquet(local_paths)


def load_dataset(dataset_name):
    """Load and normalize a dataset."""
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")
    return X_train, y_train, X_test, y_test


def get_folds(X, y, n_splits=10, random_state=None):
    folds = []
    try:
        # Try stratified split
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, val_idx in skf.split(X, y):
            folds.append((train_idx.tolist(), val_idx.tolist()))
    except ValueError:
        # Fall back to regular KFold
        print(f"StratifiedKFold failed, falling back to regular KFold with n_splits={n_splits}")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for train_idx, val_idx in kf.split(X):
            folds.append((train_idx.tolist(), val_idx.tolist()))
    return folds


class MemoryTracker:
    """Track peak RSS memory (process + children) in a background thread."""

    def __init__(self, interval=0.5):
        self.peak = 0
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        import time
        import psutil
        proc = psutil.Process(os.getpid())
        self._times = []
        self._values = []
        t0 = None
        while not self._stop.wait(self._interval):
            now = time.monotonic()
            if t0 is None:
                t0 = now
            total = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            self.peak = max(self.peak, total)
            self._times.append(now - t0)
            self._values.append(total)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join()
        return self.peak

    def plot(self, path="memory.png"):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self._times, [v / 1024**3 for v in self._values])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Memory (GB)")
        ax.set_title(f"Peak: {self.peak / 1024**3:.2f} GB")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        print(f"Memory plot saved to {path}")
        plt.close(fig)
