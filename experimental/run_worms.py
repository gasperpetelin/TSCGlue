"""Quick script to train and evaluate on the Worms dataset."""

import os
import threading
import click
import psutil
from sklearn.metrics import accuracy_score
from tscglue.models import LokyStackerV10FM
from tscglue import utils


class MemoryTracker:
    def __init__(self, interval=0.5):
        self.peak = 0
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        proc = psutil.Process(os.getpid())
        self._times = []
        self._values = []
        t0 = None
        while not self._stop.wait(self._interval):
            import time
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


@click.command()
@click.option("-d", "--dataset", default="Worms", help="Dataset name")
@click.option("-j", "--n-jobs", default=16, type=int, help="Number of parallel jobs")
def main(dataset, n_jobs):
    X_train, y_train, X_test, y_test = utils.load_dataset(dataset)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}, Classes: {len(set(y_train))}")

    tracker = MemoryTracker(interval=0.3).start()

    model = LokyStackerV10FM(random_state=270, n_repetitions=1, k_folds=10, n_jobs=n_jobs, verbose=1, feature_dtype="float32")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    peak = tracker.stop()
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Peak memory (parent+children): {peak / 1024**3:.2f} GB")

    tag = f"{dataset}_j{n_jobs}"
    tracker.plot(f"memory_{tag}.png")


if __name__ == "__main__":
    main()
