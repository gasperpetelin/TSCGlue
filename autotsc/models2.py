import gc
import os
import pickle
from time import perf_counter

import numpy as np
import polars as pl
import ray
from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.dictionary_based import WEASEL_V2, ContractableBOSS
from aeon.classification.interval_based import RSTSF, DrCIFClassifier
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.transformations.collection.interval_based import QUANTTransformer
from aeon.transformations.collection.shapelet_based import (
    RandomDilatedShapeletTransform,
)
from ray._private.internal_api import global_gc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from autotsc import utils


class SparseScaler:
    """Sparse Scaler for hydra transform (NumPy version)."""

    def __init__(self, mask=True, exponent=4):
        self.mask = mask
        self.exponent = exponent

    def fit(self, X, y=None):
        # clamp(0) → clip to minimum 0
        X = np.clip(X, 0, None)
        X = np.sqrt(X)

        # epsilon = mean((X == 0)) ** exponent + 1e-8
        zero_mask = (X == 0).astype(float)
        self.epsilon = zero_mask.mean(axis=0) ** self.exponent + 1e-8

        self.mu = X.mean(axis=0)
        self.sigma = X.std(axis=0) + self.epsilon

        return self

    def transform(self, X, y=None):
        X = np.clip(X, 0, None)
        X = np.sqrt(X)

        if self.mask:
            mask = (X != 0).astype(float)
            return ((X - self.mu) * mask) / self.sigma
        else:
            return (X - self.mu) / self.sigma

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


def generate_folds(X, y, n_splits=5, n_repetitions=5, random_state=0):
    all_folds = []
    for i in range(n_repetitions):
        folds = utils.get_folds(X, y, n_splits=n_splits, random_state=random_state + i)
        all_folds.extend(folds)
    return all_folds


class MultiScaler(BaseEstimator, TransformerMixin):
    """
    Applies different scalers to different feature groups.
    Features not in any group are ignored.

    Parameters
    ----------
    scalers : dict
        Dictionary mapping feature prefix to scaler instance.
        Example: {'hydra': SparseScaler(), 'multirocket': StandardScaler()}
    verbose : bool, default=False
        If True, print information about which features are scaled with which scaler.
    """

    def __init__(self, scalers, verbose=False):
        self.scalers = scalers
        self.verbose = verbose

    def fit(self, X, y=None):
        self.scalers_ = {}
        self.feature_groups_ = {}

        # Group columns by prefix
        for prefix, scaler in self.scalers.items():
            cols = [col for col in X.columns if col.startswith(prefix)]
            if cols:
                self.feature_groups_[prefix] = cols
                self.scalers_[prefix] = scaler
                self.scalers_[prefix].fit(X.select(cols).to_numpy())

                if self.verbose:
                    print(
                        f"[MultiScaler] {len(cols)} '{prefix}' features -> {scaler.__class__.__name__}"
                    )

        # Store output column order
        self.output_cols_ = [
            col for prefix in self.scalers.keys() for col in self.feature_groups_.get(prefix, [])
        ]

        if self.verbose:
            total_input = len(X.columns)
            total_output = len(self.output_cols_)
            ignored = total_input - total_output
            print(
                f"[MultiScaler] Total: {total_output}/{total_input} features kept, {ignored} ignored"
            )

        return self

    def transform(self, X):
        parts = []
        for prefix in self.scalers_.keys():
            if prefix in self.feature_groups_:
                cols = self.feature_groups_[prefix]
                scaled = self.scalers_[prefix].transform(X.select(cols).to_numpy())
                parts.append(scaled)

        return np.hstack(parts) if parts else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        return np.array(self.output_cols_)


class NoScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class RidgeClassifierCVIndicator(RidgeClassifierCV):
    def predict_proba(self, X):
        dists = np.zeros((X.shape[0], len(self.classes_)))
        preds = self.predict(X)
        for i in range(0, X.shape[0]):
            dists[i, np.where(self.classes_ == preds[i])] = 1
        return dists

    def _fit(self, X, y):
        with threadpool_limits(limits=1):
            return super().fit(X, y)


@ray.remote(num_cpus=1)
def train_predict(train_idx, val_idx, model_name, X, y, pipe, columns, fold_number):
    if columns is not None:
        X = pl.DataFrame(X, schema=columns)
    # print('START')
    start_train = perf_counter()
    pipe.fit(X[train_idx], y[train_idx])
    proba = pipe.predict_proba(X[val_idx])
    end_train = perf_counter()
    train_dur = end_train - start_train
    # print('END')
    return train_idx, val_idx, proba, pickle.dumps(pipe), train_dur, model_name, fold_number


@ray.remote(num_cpus=1)
def predict_proba(pipe, X, columns, model_name):
    start_train = perf_counter()
    if columns is not None:
        X = pl.DataFrame(X, schema=columns)
    prob = pipe.predict_proba(X)
    end_train = perf_counter()
    train_dur = end_train - start_train
    return prob, pipe.classes_, train_dur, model_name


class StackerV4Ray(BaseClassifier):
    def __init__(
        self,
        random_state=None,
        n_repetitions=1,
        k_folds=10,
        time_limit_in_seconds=None,
        ray_config=None,
        auto_shutdown_ray=True,
        n_jobs=-1,
    ):
        super().__init__()
        self.n_repetitions = n_repetitions
        self.k_folds = k_folds
        self.random_state = random_state
        self.cv_splits = None
        self.n_jobs = n_jobs

        self.feature_seed = np.random.default_rng(random_state)
        self.feature_transformers = []
        self.features = None
        self.predictions = None
        self.trained_models_ = None
        self.time_limit_in_seconds = time_limit_in_seconds

        self.last_repetition = False

        # Model configuration
        self.feature_models = ["multirockethydra-ridgecv", "quant-etc", "rdst-ridgecv"]
        self.series_models = ["rstsf"]  # , 'drcif', 'weasel-v2']#, 'contractable-boss']
        self.oof_models = []  # ['drcif']
        self.stacking_models = ["probability-ridgecv"]

        # Ray configuration
        self.ray_config = ray_config or {}
        self.auto_shutdown_ray = auto_shutdown_ray
        self._ray_initialized_by_us = False

        # Fallback model for when folds don't have all classes
        self._fallback_model = None

    def _ensure_ray_initialized(self, start_time=None):
        """Initialize Ray if not already running. Uses current Python environment."""
        import ray

        if not ray.is_initialized():
            # Determine number of CPUs to use
            total_cpus = os.cpu_count()
            if self.n_jobs == -1:
                num_cpus = total_cpus
            else:
                num_cpus = min(self.n_jobs, total_cpus)

            if start_time is not None:
                self._timestep_print(f"Total CPUs available: {total_cpus}", start_time)

            # Default config that uses current environment without creating copies
            default_config = {
                "num_cpus": num_cpus,
                "ignore_reinit_error": True,
                # No runtime_env - Ray will inherit current Python environment
            }
            # Merge with user-provided config
            config = {**default_config, **self.ray_config}

            if start_time is not None:
                self._timestep_print(f"Initializing Ray with {config['num_cpus']} CPUs", start_time)

            ray.init(**config)
            self._ray_initialized_by_us = True

            if start_time is not None:
                self._timestep_print("Ray initialized successfully", start_time)
        else:
            if start_time is not None:
                self._timestep_print(
                    "Ray already initialized, reusing existing instance", start_time
                )
            # We didn't initialize it, so we shouldn't shut it down
            self._ray_initialized_by_us = False

    def _shutdown_ray_if_needed(self, start_time=None):
        """Shutdown Ray if we initialized it and auto_shutdown_ray is enabled."""
        if self.auto_shutdown_ray and self._ray_initialized_by_us:
            import ray

            if ray.is_initialized():
                if start_time is not None:
                    self._timestep_print("Shutting down Ray", start_time)

                ray.shutdown()
                self._ray_initialized_by_us = False

                if start_time is not None:
                    self._timestep_print("Ray shutdown complete", start_time)

    def _get_feature_seed(self):
        return int(self.feature_seed.integers(0, 2**31 - 1, dtype=np.int32))

    def _get_transform_id(self, transform):
        """Generate a unique identifier for a transform based on its class and parameters."""
        params = transform.get_params()
        if "n_jobs" in params:
            del params["n_jobs"]
        param_strs = [f"{k}={v}" for k, v in params.items()]
        param_str = ";".join(param_strs)
        return f"{transform.__class__.__name__.lower()};{param_str}"

    def _create_training_tasks(self, model_name, current_splits, rX, rXt, ry, r_columns):
        tasks = []
        for fold_number, (train_idx, val_idx) in enumerate(current_splits):
            pipe = self.get_model(model_name, seed=self._get_feature_seed())
            data_X = rX if model_name in self.series_models else rXt
            columns = None if model_name in self.series_models else r_columns

            tasks.append(
                train_predict.remote(
                    train_idx,
                    val_idx,
                    model_name,
                    data_X,
                    ry,
                    pipe,
                    columns,
                    fold_number=fold_number,
                )
            )

        return tasks

    def _process_training_tasks(self, tasks, repetition, fit_start_time):
        model_groups = {}
        pending = tasks

        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            train_idx, val_idx, proba, pipe, train_dur, model_name, fold_number = ray.get(ready[0])

            pipe_size_mb = len(pipe) / (1024 * 1024)
            pipe = pickle.loads(pipe)

            self._timestep_print(
                f"Trained {model_name} in {train_dur:.4f}s for f-{fold_number}/r-{repetition}, size: {pipe_size_mb:.2f}MB",
                fit_start_time,
            )

            level = 0 if model_name in self.feature_models + self.series_models else 1

            predictions = self.add_probabilities0(
                proba, val_idx, pipe.classes_, model_name, level, repetition
            )
            self.predictions.extend(predictions)

            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(pipe)

        return model_groups

    def get_next_feature_transformer(self, feature_type: str):

        # implement python switch
        seed = self._get_feature_seed()
        match feature_type:
            case "multirocket":
                return MultiRocket(n_jobs=-1, random_state=seed)
            case "rdst":
                return RandomDilatedShapeletTransform(n_jobs=-1, random_state=seed)
            case "quant":
                return QUANTTransformer()
            case "hydra":
                return HydraTransformer(n_jobs=-1, random_state=seed)
            case _:
                raise ValueError(f"Unknown feature transformer type: {feature_type}")

    def get_model(self, name, seed=None):
        if name == "multirockethydra-ridgecv":
            pipe = make_pipeline(
                MultiScaler(
                    scalers={
                        "feature|hydra": SparseScaler(),
                        "feature|multirocket": StandardScaler(),
                    },
                    verbose=False,
                ),
                RidgeClassifierCVIndicator(alphas=np.logspace(-3, 3, 10)),
            )
            return pipe
        elif name == "all-ridgecv":
            pipe = make_pipeline(
                MultiScaler(
                    scalers={
                        "hydra_": SparseScaler(),
                        "multirocket_": StandardScaler(),
                        "rdst_": StandardScaler(),
                    },
                    verbose=False,
                ),
                RidgeClassifierCVIndicator(alphas=np.logspace(-3, 3, 10)),
            )
            return pipe
        elif name == "rstsf":
            return RSTSF(random_state=seed, n_jobs=1, n_estimators=100)
        elif name == "drcif":
            return DrCIFClassifier(random_state=seed, n_jobs=-1, time_limit_in_minutes=2)
        elif name == "weasel-v2":
            return WEASEL_V2(random_state=seed, n_jobs=-1)
        elif name == "contractable-boss":
            return ContractableBOSS(random_state=seed, n_jobs=-1, time_limit_in_minutes=0.1)
        elif name == "quant-etc":
            pipe = make_pipeline(
                MultiScaler(
                    scalers={
                        "feature|quant": NoScaler(),
                    },
                    verbose=False,
                ),
                ExtraTreesClassifier(
                    n_estimators=200,
                    max_features=0.1,
                    criterion="entropy",
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
            return pipe
        elif name == "rdst-ridgecv":
            pipe = make_pipeline(
                MultiScaler(
                    scalers={
                        "feature|randomdilatedshapelet": StandardScaler(),
                    },
                    verbose=False,
                ),
                RidgeClassifierCVIndicator(alphas=np.logspace(-4, 4, 20)),
            )
            return pipe
        elif name == "rdst-robustscale-ridgecv":
            pipe = make_pipeline(
                MultiScaler(
                    scalers={
                        "rdst_": RobustScaler(),
                    },
                    verbose=False,
                ),
                RidgeClassifierCVIndicator(alphas=np.logspace(-4, 4, 20)),
            )
            return pipe
        elif name == "catch22-quant-et":
            pipe = make_pipeline(
                MultiScaler(
                    scalers={
                        "catch22_": NoScaler(),
                        "quant_": NoScaler(),
                    },
                    verbose=False,
                ),
                ExtraTreesClassifier(
                    n_estimators=200,
                    max_features=0.1,
                    criterion="entropy",
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
            return pipe
        elif name == "probability-linear-svc":
            pipe = make_pipeline(
                MultiScaler(
                    scalers={
                        "proba_": StandardScaler(),
                    },
                    verbose=False,
                ),
                SVC(kernel="linear", probability=True, random_state=seed),
            )
            return pipe
        elif name == "probability-et":
            pipe = make_pipeline(
                MultiScaler(
                    scalers={
                        "proba_": StandardScaler(),
                    },
                    verbose=False,
                ),
                ExtraTreesClassifier(
                    n_estimators=500,
                    # max_features=0.3,
                    # criterion="entropy",
                    random_state=seed,
                    n_jobs=-1,
                    bootstrap=True,
                ),
            )
            return pipe
        elif name == "probability-ridgecv":
            pipe = make_pipeline(
                MultiScaler(
                    scalers={
                        "probability|model0": StandardScaler(),
                    },
                    verbose=False,
                ),
                RidgeClassifierCVIndicator(alphas=np.logspace(-3, 3, 20)),
            )
            return pipe
        elif name == "probability-rf":
            pipe = make_pipeline(
                MultiScaler(
                    scalers={
                        "proba_": StandardScaler(),
                    },
                    verbose=False,
                ),
                RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
            )
            return pipe
        else:
            raise ValueError(f"Unknown model name: {name}")

    def _timestep_print(self, message: str, start_time: float):
        elapsed_time = perf_counter() - start_time
        print(f"[{elapsed_time:.2f}s] {message}")

    def _print_ray_memory_summary(self, start_time: float):
        from ray._private.internal_api import memory_summary

        summary = memory_summary(stats_only=True)
        elapsed_time = perf_counter() - start_time
        # print(f'[{elapsed_time:.2f}s] Ray memory summary: {summary}')
        summary = summary.replace("\n", " ").replace(
            "--- Aggregate object store stats across all nodes ---", ""
        )
        self._timestep_print(f"Ray memory summary: {summary}", start_time)

    def _train_fallback_model(self, X, y, fit_start_time):
        """Train fallback MultiRocketHydraClassifier model."""
        self._timestep_print(
            "Training fallback MultiRocketHydraClassifier model...",
            fit_start_time,
        )
        self._fallback_model = MultiRocketHydraClassifier(
            n_jobs=self.n_jobs, random_state=self.random_state
        )
        self._fallback_model.fit(X, y)
        self._timestep_print(
            "Fallback MultiRocketHydraClassifier trained successfully", fit_start_time
        )
        self.best_model = "fallback"

    def _feature_calc(self, X, fit_start_time):
        multirocket_start_time = perf_counter()
        self.add_features(feature_type="multirocket", X=X)
        multirocket_durration = perf_counter() - multirocket_start_time
        self._timestep_print(
            f"Computed MultiRocket features in {multirocket_durration:.2f}s", fit_start_time
        )

        hydra_start_time = perf_counter()
        self.add_features(feature_type="hydra", X=X)
        hydra_durration = perf_counter() - hydra_start_time
        self._timestep_print(f"Computed Hydra features in {hydra_durration:.2f}s", fit_start_time)

        rdst_start_time = perf_counter()
        self.add_features(feature_type="rdst", X=X)
        rdst_durration = perf_counter() - rdst_start_time
        self._timestep_print(f"Computed RDST features in {rdst_durration:.2f}s", fit_start_time)

    def _fit(self, X, y):
        fit_start_time = perf_counter()
        self._timestep_print("Starting fitting", fit_start_time)

        # Check if each class has at least 2 samples (required for k-fold CV)
        unique_classes, class_counts = np.unique(y, return_counts=True)
        self._timestep_print(
            f"Class distribution: {dict(zip(unique_classes, class_counts))}", fit_start_time
        )

        min_class_count = np.min(class_counts)
        if min_class_count < 2:
            self._timestep_print(
                f"At least one class has only {min_class_count} sample(s). K-fold CV not possible. Using fallback.",
                fit_start_time,
            )
            self._train_fallback_model(X, y, fit_start_time)
            return

        # Ensure Ray is initialized before starting
        self._ensure_ray_initialized(fit_start_time)

        self.predictions = []
        self.trained_models_ = []

        if self.cv_splits is None:
            self.cv_splits = []

        quant_start_time = perf_counter()
        self.add_features(feature_type="quant", X=X)
        quant_durration = perf_counter() - quant_start_time
        self._timestep_print(f"Computed QUANT features in {quant_durration:.2f}s", fit_start_time)

        rX = ray.put(X)
        ry = ray.put(y)
        self._print_ray_memory_summary(fit_start_time)

        for repetition in range(self.n_repetitions):
            if self.last_repetition:
                break

            self._timestep_print(f"Starting repetition {repetition}", fit_start_time)

            self._feature_calc(X, fit_start_time)

            Xt = self.get_Xt()
            rXt = ray.put(Xt.to_numpy())
            r_columns = ray.put(Xt.columns)
            current_splits = generate_folds(
                X, y, n_splits=self.k_folds, n_repetitions=1, random_state=self._get_feature_seed()
            )

            self._print_ray_memory_summary(fit_start_time)

            tasks = []
            for model_name in self.feature_models + self.series_models:
                # check if model trainign should be terminated due to time limit
                # skip only nonstacking models
                elapsed_time = perf_counter() - fit_start_time
                if (
                    self.time_limit_in_seconds is not None
                    and elapsed_time > self.time_limit_in_seconds
                ):
                    self.last_repetition = True
                    if model_name not in self.stacking_models:
                        self._timestep_print(
                            f"Skipping training of model {model_name} due to time limit",
                            fit_start_time,
                        )
                        continue

                model_tasks = self._create_training_tasks(
                    model_name, current_splits, rX, rXt, ry, r_columns
                )
                tasks.extend(model_tasks)

                self._timestep_print(f"Tasks queued for model {model_name}", fit_start_time)

            model_groups = self._process_training_tasks(tasks, repetition, fit_start_time)

            for model_name, model_group in model_groups.items():
                self.trained_models_.append((model_name, model_group))

            Xt = self.get_Xt()
            for model_name in self.feature_models + self.series_models:
                prob_columns = [col for col in Xt.columns if model_name in col]
                agg_probs = Xt.select(prob_columns)
                oof_probas = agg_probs.to_numpy()
                oof_pred_indices = np.argmax(oof_probas, axis=1)
                oof_preds = self.classes_[oof_pred_indices]
                oof_acc = accuracy_score(y, oof_preds)
                self._timestep_print(f"OOF acc for model {model_name}: {oof_acc}", fit_start_time)

            self._print_ray_memory_summary(fit_start_time)
            del rXt, tasks, r_columns
            gc.collect()
            global_gc()
            self._print_ray_memory_summary(fit_start_time)

            Xt = self.get_Xt()
            rXt = ray.put(Xt.to_numpy())
            r_columns = ray.put(Xt.columns)

            for model_name in self.stacking_models:
                model_group = []

                # check if model trainign should be terminated due to time limit
                # skip only nonstacking models
                elapsed_time = perf_counter() - fit_start_time
                if (
                    self.time_limit_in_seconds is not None
                    and elapsed_time > self.time_limit_in_seconds
                ):
                    self.last_repetition = True
                    if model_name not in self.stacking_models:
                        self._timestep_print(
                            f"Skipping training of model {model_name} due to time limit",
                            fit_start_time,
                        )
                        continue

                self._timestep_print("Data put called", fit_start_time)

                tasks = self._create_training_tasks(
                    model_name, current_splits, rX, rXt, ry, r_columns
                )

                self._timestep_print(f"Tasks created for model {model_name}", fit_start_time)

                model_groups = self._process_training_tasks(tasks, repetition, fit_start_time)

                for model_name, model_group in model_groups.items():
                    self.trained_models_.append((model_name, model_group))

                # get updated Xt
                Xt = self.get_Xt()
                prob_columns = [col for col in Xt.columns if model_name in col]
                agg_probs = Xt.select(prob_columns)
                oof_probas = agg_probs.to_numpy()
                oof_pred_indices = np.argmax(oof_probas, axis=1)
                oof_preds = self.classes_[oof_pred_indices]
                oof_acc = accuracy_score(y, oof_preds)

                self._timestep_print(f"OOF acc for model {model_name}: {oof_acc}", fit_start_time)

            for model_name in self.oof_models:
                pipe = self.get_model(model_name, seed=self._get_feature_seed())
                start_train = perf_counter()
                oof_probas = pipe.fit_predict_proba(X, y)
                end_train = perf_counter()
                train_dur = end_train - start_train
                self._timestep_print(
                    f"Trained OOF model {model_name} in {train_dur:.4f}s for r-{repetition}",
                    fit_start_time,
                )

                oof_pred_indices = np.argmax(oof_probas, axis=1)
                oof_preds = self.classes_[oof_pred_indices]
                oof_acc = accuracy_score(y, oof_preds)
                self._timestep_print(
                    f"OOF acc for model {model_name} after repetition {repetition}: {oof_acc}",
                    fit_start_time,
                )

            self._timestep_print(f"Completed repetition {repetition}", fit_start_time)
        self._timestep_print("Completed all repetitions", fit_start_time)
        self.best_model = "probability-ridgecv"

        self._print_ray_memory_summary(fit_start_time)
        del X, rX, rXt, tasks, r_columns, ry, y
        gc.collect()
        global_gc()
        self._print_ray_memory_summary(fit_start_time)

        # Shutdown Ray if we initialized it and auto_shutdown is enabled
        self._shutdown_ray_if_needed(fit_start_time)

    def combine_features_and_predictions(self, features, predictions):
        if len(predictions) == 0:
            return features
        df = (
            pl.DataFrame(predictions)
            .pivot(
                values="probability",
                index="index",
                on=["level", "model", "class"],
                aggregate_function="mean",
            )
            .sort("index")
        )
        rename_dict = {}
        for c in df.columns[1:]:
            t = tuple([v.replace("{", "").replace("}", "").replace('"', "") for v in c.split(",")])
            level, model_name, prob_class = t
            rename_dict[c] = f"probability|model{level}_{model_name}_class_{prob_class}"

        df = df.rename(rename_dict)

        return (
            features.with_row_index("index")
            .join(df, on="index", how="full")
            .sort("index")
            .drop("index", "index_right")
        )

    def add_probabilities0(self, probas, val_idx, classes, model_name, level, repetition):
        predictions = []
        for idx_in_fold, p in zip(val_idx, probas):
            for scls, prob in zip(classes, p):
                d = {
                    "index": idx_in_fold,
                    "model": model_name,
                    "level": level,
                    "repetition": repetition,
                    "class": scls.item(),
                    "probability": prob.item(),
                }
                predictions.append(d)
        return predictions

    def add_probabilities(self, probas, classes, model_name, level, repetition=None):
        predictions = []
        for idx, p in enumerate(probas):
            for scls, prob in zip(classes, p):
                d = {
                    "index": idx,
                    "model": model_name,
                    "level": level,
                    "class": scls.item(),
                    "probability": prob.item(),
                }
                if repetition is not None:
                    d["repetition"] = repetition
                predictions.append(d)
        return predictions

    def _predict_one_model(self, model, model_name, X, Xt_):
        start_predict = perf_counter()
        if model_name in self.series_models:
            proba = model.predict_proba(X)
        else:
            proba = model.predict_proba(Xt_)
        end_predict = perf_counter()
        predict_dur = end_predict - start_predict
        return proba, model.classes_, predict_dur

    def predict_proba_per_model(self, X):
        predict_start_time = perf_counter()
        self._timestep_print("Starting prediction", predict_start_time)

        # Ensure Ray is initialized before prediction
        self._ensure_ray_initialized(predict_start_time)
        return_dict = {}
        Xt = self.compute_features(X, predict_start_time)
        self._timestep_print("Computed all features for prediction", predict_start_time)
        predictions = []

        rX = ray.put(X)
        Xt_ = self.combine_features_and_predictions(Xt, predictions)
        rXt = ray.put(Xt_.to_numpy())
        r_columns = ray.put(Xt_.columns)

        tasks = []
        for model_name, model_group in self.trained_models_:
            if model_name in self.stacking_models:
                continue

            for model in model_group:
                # Select data source based on model type
                data_X = rX if model_name in self.series_models else rXt
                columns = None if model_name in self.series_models else r_columns

                tasks.append(predict_proba.remote(model, data_X, columns, model_name=model_name))

        pending = tasks
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            proba, classes_, predict_dur, model_name = ray.get(ready[0])

            self._timestep_print(
                f"Predicted probabilities with {model_name} in {predict_dur:.4f}s",
                predict_start_time,
            )

            level = 0 if model_name in self.feature_models + self.series_models else 1
            pred_list = self.add_probabilities(proba, classes_, model_name, level)
            predictions.extend(pred_list)

        tasks = []
        Xt_ = self.combine_features_and_predictions(Xt, predictions)
        rXt = ray.put(Xt_.to_numpy())

        for model_name, model_group in self.trained_models_[::-1]:
            if model_name not in self.stacking_models:
                continue

            for model in model_group:
                # Select data source based on model type
                if model_name in self.series_models:
                    data_X = rX
                    columns = None
                else:
                    data_X = rXt
                    columns = ray.put(Xt_.columns)

                tasks.append(predict_proba.remote(model, data_X, columns, model_name=model_name))

            pending = tasks
            while pending:
                ready, pending = ray.wait(pending, num_returns=1)
                proba, classes_, predict_dur, model_name = ray.get(ready[0])

                self._timestep_print(
                    f"Predicted probabilities with {model_name} in {predict_dur:.4f}s",
                    predict_start_time,
                )

                level = 0 if model_name in self.feature_models + self.series_models else 1
                pred_list = self.add_probabilities(proba, classes_, model_name, level)
                predictions.extend(pred_list)

        df = self.combine_features_and_predictions(Xt, predictions)
        for model_name, _ in self.trained_models_:
            prob_columns = [col for col in df.columns if model_name in col]
            agg_probs = df.select(prob_columns)
            return_dict[model_name] = agg_probs.to_numpy()

        del rX, rXt, r_columns, tasks
        gc.collect()
        global_gc()
        self._print_ray_memory_summary(predict_start_time)

        # Shutdown Ray if we initialized it and auto_shutdown is enabled
        self._shutdown_ray_if_needed(predict_start_time)

        return return_dict

    def _predict_proba(self, X):
        if self._fallback_model is not None:
            return self._fallback_model.predict_proba(X)
        return self.predict_proba_per_model(X)[self.best_model]

    def _predict(self, X):
        if self._fallback_model is not None:
            return self._fallback_model.predict(X)
        probas = self._predict_proba(X)
        predicted_indices = np.argmax(probas, axis=1)
        return self.classes_[predicted_indices]

    def add_features(self, feature_type: str, X: np.ndarray):
        transform = self.get_next_feature_transformer(feature_type=feature_type)
        transform.fit(X)
        X_t = transform.transform(X)
        transform_id = self._get_transform_id(transform)
        schema = ["feature|" + transform_id + ";index=" + str(i) for i in range(X_t.shape[1])]
        feature_df = pl.DataFrame(X_t, schema=schema)
        self.feature_transformers.append(transform)
        if self.features is None:
            self.features = feature_df
        else:
            self.features = pl.concat([self.features, feature_df], how="horizontal")

    def compute_features(self, X: np.ndarray, start_time):
        feature_dfs = []
        for transform in self.feature_transformers:
            transform_start = perf_counter()
            X_t = transform.transform(X)
            transform_duration = perf_counter() - transform_start

            transform_id = self._get_transform_id(transform)

            self._timestep_print(
                f"Computed {transform.__class__.__name__} features in {transform_duration:.2f}s",
                start_time
            )

            schema = ["feature|" + transform_id + ";index=" + str(i) for i in range(X_t.shape[1])]
            feature_df = pl.DataFrame(X_t, schema=schema)
            feature_dfs.append(feature_df)
        return pl.concat(feature_dfs, how="horizontal")

    def get_Xt(self):
        return self.combine_features_and_predictions(self.features, self.predictions)
