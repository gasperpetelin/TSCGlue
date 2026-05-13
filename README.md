# TSCGlue

Code accompanying the paper submission. TSCGlue is an ensemble classifier for univariate time series that combines multiple base classifiers via stacking.

## Setup

```bash
make setup
```

## Implementation

The TSCGlue model is implemented in `tscglue/models.py` as the `TSCGlue` class.

## Running experiments

All scripts are in `experimental/`. Experiments use SLURM array jobs — each array index corresponds to one fold (0–29 over 30 resamples).

**Run all models on all datasets:**
```bash
cd experimental
sbatch run_stacking2.slurm
```

**Run a specific model:**
```bash
sbatch run_stacking2.slurm -m <model-name>
```

**Run a specific model on a specific dataset:**
```bash
sbatch run_stacking2.slurm -m <model-name> -d <dataset-name>
```

**List all available models and datasets:**
```bash
uv run run_stacking2.py --list-models
uv run run_stacking2.py --list-datasets
```

**Timing benchmarks:**
```bash
sbatch run_timing.slurm -m <model-name>
# Control parallelism: N_JOBS=1 sbatch run_timing.slurm -m <model-name>
```

Results are saved to `performance-benchmarking/` on disk (pass `--storage s3` to write to S3 instead).

## Quick example

```python
from aeon.datasets import load_classification
from sklearn.metrics import accuracy_score
from tscglue.models import TSCGlue

X_train, y_train = load_classification("ArrowHead", split="train")
X_test, y_test = load_classification("ArrowHead", split="test")

model = TSCGlue(random_state=0, n_jobs=4)
model.fit(X_train, y_train)
print(accuracy_score(y_test, model.predict(X_test)))
```

## Figures and analysis

`notebooks/paper-figures.ipynb` contains all figures and statistical analysis used in the paper, including critical difference diagrams, multi-comparison matrices, ablation studies, and timing plots.
