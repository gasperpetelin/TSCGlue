import json
import os
import tempfile

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
import numpy as np
import polars as pl
import ray
import multiprocessing
from multiprocessing import Process, Queue as MPQueue, Manager
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
from joblib import Parallel, delayed, dump, load
from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.dictionary_based import WEASEL_V2, ContractableBOSS
from aeon.classification.interval_based import RSTSF, DrCIFClassifier, QUANTClassifier
from aeon.classification.shapelet_based import RDSTClassifier
from aeon.pipeline import make_pipeline as aeon_make_pipeline
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.transformations.collection.interval_based import QUANTTransformer
from aeon.transformations.collection.shapelet_based import (
    RandomDilatedShapeletTransform,
)
import pickle
from ray.util.queue import Queue
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score

# from run_stacking import generate_folds, SparseScaler, MultiScaler, RidgeClassifierCVIndicator, NoScaler, StackerV4
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from autotsc import old_models, transformers, utils


def save_mmap_data(X: np.ndarray, Xt: pl.DataFrame, y: np.ndarray, tmpdir: str) -> tuple[str, str, str]:
    X_path = f"{tmpdir}/X.npy"
    Xt_path = f"{tmpdir}/Xt.ipc"
    y_path = f"{tmpdir}/y.npy"
    np.save(X_path, X)
    np.save(y_path, y)
    Xt.write_ipc(Xt_path)
    return X_path, Xt_path, y_path


def load_mmap_data(X_path: str, Xt_path: str, y_path: str) -> tuple[np.ndarray, pl.DataFrame, np.ndarray]:
    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')
    Xt = pl.read_ipc(Xt_path, memory_map=True)
    return X, Xt, y


@ray.remote(num_cpus=1)
def ray_run_predict_proba(model, X):
    start_time = perf_counter()
    proba = model.predict_proba(X)
    pred_time = perf_counter() - start_time
    return proba, pred_time


class StackerV4(BaseClassifier):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, time_limit_in_seconds=None):
        super().__init__()
        self.n_repetitions = n_repetitions
        self.k_folds = k_folds
        self.random_state = random_state
        self.cv_splits = None

        self.feature_seed = np.random.default_rng(random_state)
        self.feature_transformers = []
        self.features = None
        self.predictions = None
        self.trained_models_ = None
        self.time_limit_in_seconds = time_limit_in_seconds

        self.last_repetition = False
        self._fallback_model = None

    def _get_feature_seed(self):
        return int(self.feature_seed.integers(0, 2**31 - 1, dtype=np.int32))

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

    def get_model(self, name, seed=None, n_jobs=1):
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
            return RSTSF(random_state=seed, n_jobs=n_jobs, n_estimators=100)
        elif name == "drcif":
            return DrCIFClassifier(random_state=seed, n_jobs=n_jobs, time_limit_in_minutes=2)
        elif name == "weasel-v2":
            return WEASEL_V2(random_state=seed, n_jobs=n_jobs)
        elif name == "contractable-boss":
            return ContractableBOSS(random_state=seed, n_jobs=n_jobs, time_limit_in_minutes=0.1)
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
                    n_jobs=n_jobs,
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
                    n_jobs=n_jobs,
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
                    n_jobs=n_jobs,
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
                RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=n_jobs),
            )
            return pipe
        else:
            raise ValueError(f"Unknown model name: {name}")

    def _fit(self, X, y):
        fit_start_time = perf_counter()
        try:
            self._fit_internal(X, y, fit_start_time)
        except Exception as e:
            print(f"[{perf_counter() - fit_start_time:.4f}s] Training failed with exception: {e}")
            print(f"[{perf_counter() - fit_start_time:.4f}s] Falling back to MultiRocketHydraClassifier")
            self._fallback_model = MultiRocketHydraClassifier(random_state=self.random_state)
            self._fallback_model.fit(X, y)
            print(f"[{perf_counter() - fit_start_time:.4f}s] Fallback model trained successfully")

    def _fit_internal(self, X, y, fit_start_time):
        self.predictions = []
        self.trained_models_ = []

        self.feature_models = ["multirockethydra-ridgecv", "quant-etc", "rdst-ridgecv"]
        self.series_models = ["rstsf"]  # , 'drcif', 'weasel-v2']#, 'contractable-boss']
        self.oof_models = []  # ['drcif']
        self.stacking_models = ["probability-ridgecv"]

        if self.cv_splits is None:
            self.cv_splits = []
        print(f"[{perf_counter() - fit_start_time:.4f}s] Starting fitting")

        quant_start_time = perf_counter()
        self.add_features(feature_type="quant", X=X)
        quant_durration = perf_counter() - quant_start_time
        print(
            f"[{perf_counter() - fit_start_time:.4f}s] Computed QUANT features in {quant_durration:.4f}s"
        )

        for repetition in range(self.n_repetitions):
            if self.last_repetition:
                break

            print(f"[{perf_counter() - fit_start_time:.4f}s] Starting repetition {repetition}")

            multirocket_start_time = perf_counter()
            self.add_features(feature_type="multirocket", X=X)
            multirocket_durration = perf_counter() - multirocket_start_time
            print(
                f"[{perf_counter() - fit_start_time:.4f}s] Computed MultiRocket features in {multirocket_durration:.4f}s"
            )

            hydra_start_time = perf_counter()
            self.add_features(feature_type="hydra", X=X)
            hydra_durration = perf_counter() - hydra_start_time
            print(
                f"[{perf_counter() - fit_start_time:.4f}s] Computed Hydra features in {hydra_durration:.4f}s"
            )

            rdst_start_time = perf_counter()
            self.add_features(feature_type="rdst", X=X)
            rdst_durration = perf_counter() - rdst_start_time
            print(
                f"[{perf_counter() - fit_start_time:.4f}s] Computed RDST features in {rdst_durration:.4f}s"
            )

            current_splits = generate_folds(
                X, y, n_splits=self.k_folds, n_repetitions=1, random_state=self._get_feature_seed()
            )
            for model_name in self.feature_models + self.series_models + self.stacking_models:
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
                        print(
                            f"[{perf_counter() - fit_start_time:.4f}s] Skipping training of model {model_name} due to time limit"
                        )
                        continue

                Xt = self.get_Xt()
                for fold_number, (train_idx, val_idx) in enumerate(current_splits):
                    model_seed = self._get_feature_seed()
                    pipe = self.get_model(model_name, seed=model_seed)

                    start_train = perf_counter()
                    if model_name in self.series_models:
                        pipe.fit(X[train_idx], y[train_idx])
                        proba = pipe.predict_proba(X[val_idx])
                    else:
                        pipe.fit(Xt[train_idx], y[train_idx])
                        proba = pipe.predict_proba(Xt[val_idx])
                    end_train = perf_counter()
                    train_dur = end_train - start_train

                    print(
                        f"[{perf_counter() - fit_start_time:.4f}s] Trained {model_name} in {train_dur:.4f}s for f-{fold_number}/r-{repetition}"
                    )

                    level = 0 if model_name in self.feature_models + self.series_models else 1

                    for idx, p in zip(val_idx, proba):
                        for scls, prob in zip(pipe.classes_, p):
                            d = {
                                "index": idx,
                                "model": model_name,
                                "repetition": repetition,
                                "level": level,
                                "class": scls.item(),
                                "probability": prob.item(),
                            }
                            self.predictions.append(d)
                    model_group.append(pipe)

                self.trained_models_.append((model_name, model_group))

                # get updated Xt
                Xt = self.get_Xt()
                prob_columns = [col for col in Xt.columns if model_name in col]
                agg_probs = Xt.select(prob_columns)
                oof_probas = agg_probs.to_numpy()
                oof_pred_indices = np.argmax(oof_probas, axis=1)
                oof_preds = self.classes_[oof_pred_indices]
                oof_acc = accuracy_score(y, oof_preds)

                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] OOF acc for model {model_name}: {oof_acc}"
                )

            for model_name in self.oof_models:
                pipe = self.get_model(model_name)
                start_train = perf_counter()
                oof_probas = pipe.fit_predict_proba(X, y)
                end_train = perf_counter()
                train_dur = end_train - start_train
                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] Trained OOF model {model_name} in {train_dur:.4f}s for r-{repetition}"
                )

                oof_pred_indices = np.argmax(oof_probas, axis=1)
                oof_preds = self.classes_[oof_pred_indices]
                oof_acc = accuracy_score(y, oof_preds)
                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] OOF acc for model {model_name} after repetition {repetition}: {oof_acc}"
                )

            print(f"[{perf_counter() - fit_start_time:.4f}s] Completed repetition {repetition}")
        print(f"[{perf_counter() - fit_start_time:.4f}s] Completed all repetitions")
        self.best_model = "probability-ridgecv"

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

    def add_probabilities(self, probas, classes, model_name, level):
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
        if self._fallback_model is not None:
            return {"fallback": self._fallback_model.predict_proba(X)}

        from sklearn.utils.parallel import Parallel, delayed

        predict_start_time = perf_counter()
        print(f"[{perf_counter() - predict_start_time:.4f}s] Starting prediction")
        return_dict = {}
        Xt = self.compute_features(X)
        print(f"[{perf_counter() - predict_start_time:.4f}s] Computed features for prediction")
        predictions = []
        for model_name, model_group in self.trained_models_:
            Xt_ = self.combine_features_and_predictions(Xt, predictions)

            results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(self._predict_one_model)(model, model_name, X, Xt_) for model in model_group
            )
            for proba, classes_, predict_dur in results:
                print(
                    f"[{perf_counter() - predict_start_time:.4f}s] Predicted probabilities with {model_name} in {predict_dur:.4f}s"
                )

                level = 0 if model_name in self.feature_models + self.series_models else 1

                pred_list = self.add_probabilities(proba, classes_, model_name, level)
                predictions.extend(pred_list)

        df = self.combine_features_and_predictions(Xt, predictions)
        for model_name, _ in self.trained_models_:
            prob_columns = [col for col in df.columns if model_name in col]
            agg_probs = df.select(prob_columns)
            return_dict[model_name] = agg_probs.to_numpy()
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
        params = transform.get_params()
        if "n_jobs" in params:
            del params["n_jobs"]
        param_strs = [f"{k}={v}" for k, v in params.items()]
        param_str = ";".join(param_strs)
        transform_id = f"{transform.__class__.__name__.lower()};{param_str}"
        schema = ["feature|" + transform_id + ";index=" + str(i) for i in range(X_t.shape[1])]
        feature_df = pl.DataFrame(X_t, schema=schema)
        self.feature_transformers.append(transform)
        if self.features is None:
            self.features = feature_df
        else:
            self.features = pl.concat([self.features, feature_df], how="horizontal")

    def compute_features(self, X: np.ndarray):
        feature_dfs = []
        for transform in self.feature_transformers:
            X_t = transform.transform(X)
            params = transform.get_params()
            if "n_jobs" in params:
                del params["n_jobs"]
            param_strs = [f"{k}={v}" for k, v in params.items()]
            param_str = ";".join(param_strs)
            transform_id = f"{transform.__class__.__name__.lower()};{param_str}"
            schema = ["feature|" + transform_id + ";index=" + str(i) for i in range(X_t.shape[1])]
            feature_df = pl.DataFrame(X_t, schema=schema)
            feature_dfs.append(feature_df)
        return pl.concat(feature_dfs, how="horizontal")

    def get_Xt(self):
        return self.combine_features_and_predictions(self.features, self.predictions)


def get_model(name, seed=None, n_jobs=1):
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
        return RSTSF(random_state=seed, n_jobs=n_jobs, n_estimators=100)
    elif name == "drcif":
        return DrCIFClassifier(random_state=seed, n_jobs=n_jobs, time_limit_in_minutes=2)
    elif name == "weasel-v2":
        return WEASEL_V2(random_state=seed, n_jobs=n_jobs)
    elif name == "contractable-boss":
        return ContractableBOSS(random_state=seed, n_jobs=n_jobs, time_limit_in_minutes=0.1)
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
                n_jobs=n_jobs,
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
                n_jobs=n_jobs,
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
                n_jobs=n_jobs,
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
            RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=n_jobs),
        )
        return pipe
    else:
        raise ValueError(f"Unknown model name: {name}")

@ray.remote(num_cpus=1)
def train_predict(worker_id, run_queue, result_queue, X, Xt, y, X_columns):
    Xt = pl.DataFrame(Xt, schema=X_columns) # Optimize this out
    while True:
        run_params = run_queue.get()
        if run_params is None:
            break

        fold_number, model_name, use_Xt, train_idx, val_idx, model_seed = run_params
        pipe = get_model(model_name, seed=model_seed)
        start_train = perf_counter()

        if use_Xt:
            pipe.fit(Xt[train_idx], y[train_idx])
            proba = pipe.predict_proba(Xt[val_idx])
        else:
            pipe.fit(X[train_idx], y[train_idx])
            proba = pipe.predict_proba(X[val_idx])

        end_train = perf_counter()
        train_dur = end_train - start_train
        rtrn = (worker_id, train_idx, val_idx, proba, pickle.dumps(pipe), train_dur, model_name, fold_number)
        result_queue.put(rtrn)


@ray.remote
def predict_worker(worker_id, run_queue, result_queue, X, Xt, Xt_columns):
    """Ray remote worker for prediction, similar to train_predict."""
    Xt = pl.DataFrame(Xt, schema=Xt_columns)
    while True:
        run_params = run_queue.get()
        if run_params is None:
            break

        model_name, pickle_pipe, use_series = run_params
        pipe = pickle.loads(pickle_pipe)
        start_predict = perf_counter()

        if use_series:
            proba = pipe.predict_proba(X)
        else:
            proba = pipe.predict_proba(Xt)

        end_predict = perf_counter()
        predict_dur = end_predict - start_predict
        rtrn = (worker_id, proba, pipe.classes_, predict_dur, model_name)
        result_queue.put(rtrn)


class FastStackerV4(BaseClassifier):
    """Copy of StackerV4 for parallelization experiments."""
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, time_limit_in_seconds=None, n_jobs=1,
                 isolated_ray=True, ray_port=None):
        super().__init__()
        self.n_repetitions = n_repetitions
        self.k_folds = k_folds
        self.random_state = random_state
        self.cv_splits = None
        self.n_jobs = n_jobs
        self.isolated_ray = isolated_ray
        self.ray_port = ray_port

        self.feature_seed = np.random.default_rng(random_state)
        self.feature_transformers = []
        self.features = None
        self.predictions = None
        self.trained_models_ = None
        self.time_limit_in_seconds = time_limit_in_seconds

        self.last_repetition = False

    def _get_available_port(self):
        """Find a random available port in range 60000-65535."""
        import socket
        import random
        for _ in range(100):
            port = random.randint(60000, 65535)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("Could not find an available port in range 60000-65535")

    def _init_ray(self):
        """Initialize Ray, optionally as an isolated instance with unique temp dir."""
        if self.isolated_ray:
            # Start a completely separate Ray cluster with unique temp directory
            # This allows multiple FastStackerV4 instances to run in parallel
            if self.ray_port is None:
                self.ray_port = self._get_available_port()
            ray_tmp = os.path.expanduser(f"~/ray/p_{self.ray_port}")
            ray_spill = os.path.expanduser(f"~/ray/s_{self.ray_port}")
            os.makedirs(ray_tmp, exist_ok=True)
            os.makedirs(ray_spill, exist_ok=True)
            print(f"Starting isolated Ray on port {self.ray_port} with temp_dir={ray_tmp}")
            ray.init(
                num_cpus=self.n_jobs,
                ignore_reinit_error=False,
                include_dashboard=False,
                _temp_dir=ray_tmp,
                _system_config={
                    "object_spilling_config": json.dumps({
                        "type": "filesystem",
                        "params": {"directory_path": ray_spill}
                    }),
                    "memory_monitor_refresh_ms": 0,
                },
            )
        else:
            print("Starting shared Ray instance")
            ray.init(
                num_cpus=self.n_jobs,
                ignore_reinit_error=True,
            )

    def _get_feature_seed(self):
        return int(self.feature_seed.integers(0, 2**31 - 1, dtype=np.int32))

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

    def _fit(self, X, y): # FastStackerv4 fit
        fit_start_time = perf_counter()

        self.predictions = []
        self.trained_models_ = []

        self.feature_models = ["multirockethydra-ridgecv", "quant-etc", "rdst-ridgecv"]
        self.series_models = ["rstsf"]  # , 'drcif', 'weasel-v2']#, 'contractable-boss']
        self.oof_models = []  # ['drcif']
        self.stacking_models = ["probability-ridgecv"]

        if self.cv_splits is None:
            self.cv_splits = []
        print(f"[{perf_counter() - fit_start_time:.4f}s] Starting fitting")

        quant_start_time = perf_counter()
        self.add_features(feature_type="quant", X=X)
        quant_durration = perf_counter() - quant_start_time
        print(
            f"[{perf_counter() - fit_start_time:.4f}s] Computed QUANT features in {quant_durration:.4f}s"
        )

        self._init_ray()

        for repetition in range(self.n_repetitions):
            print(f"[{perf_counter() - fit_start_time:.4f}s] Starting repetition {repetition}")

            multirocket_start_time = perf_counter()
            self.add_features(feature_type="multirocket", X=X)
            multirocket_durration = perf_counter() - multirocket_start_time
            print(
                f"[{perf_counter() - fit_start_time:.4f}s] Computed MultiRocket features in {multirocket_durration:.4f}s"
            )

            hydra_start_time = perf_counter()
            self.add_features(feature_type="hydra", X=X)
            hydra_durration = perf_counter() - hydra_start_time
            print(
                f"[{perf_counter() - fit_start_time:.4f}s] Computed Hydra features in {hydra_durration:.4f}s"
            )

            rdst_start_time = perf_counter()
            self.add_features(feature_type="rdst", X=X)
            rdst_durration = perf_counter() - rdst_start_time
            print(
                f"[{perf_counter() - fit_start_time:.4f}s] Computed RDST features in {rdst_durration:.4f}s"
            )

            current_splits = generate_folds(
                X, y, n_splits=self.k_folds, n_repetitions=1, random_state=self._get_feature_seed()
            )


            rX = ray.put(X)
            ry = ray.put(y)
            run_queue = Queue()
            result_queue = Queue()

            Xt = self.get_Xt()
            rXt = ray.put(Xt.to_numpy())
            rXt_columns = ray.put(Xt.columns)
            tasks = []

            print(f"[{perf_counter() - fit_start_time:.4f}s] Data added to ray storage")

            model_count = 0
            for model_name in self.feature_models + self.series_models:
                for fold_number, (train_idx, val_idx) in enumerate(current_splits):
                    use_Xt = model_name not in self.series_models
                    model_seed = self._get_feature_seed()
                    run_queue.put((fold_number, model_name, use_Xt, train_idx, val_idx, model_seed))
                    model_count += 1

            for _ in range(self.n_jobs):
                run_queue.put(None)

            print(f"[{perf_counter() - fit_start_time:.4f}s] Starting RAY training with {self.n_jobs} workers for {model_count} models")

            for i in range(self.n_jobs):
                tasks.append(train_predict.remote(i, run_queue, result_queue, rX, rXt, ry, rXt_columns))

            model_groups = {}
            for _ in range(model_count):
                worker_id, train_idx, val_idx, proba, pickle_pipe, train_dur, model_name, fold_number = result_queue.get()
                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] Trained {model_name} on {worker_id} in {train_dur:.4f}s for f-{fold_number}/r-{repetition}"
                )
                pipe = pickle.loads(pickle_pipe)

                level = 0 if model_name in self.feature_models + self.series_models else 1
                for idx, p in zip(val_idx, proba):
                    for scls, prob in zip(pipe.classes_, p):
                        d = {
                            "index": idx,
                            "model": model_name,
                            "repetition": repetition,
                            "level": level,
                            "class": scls.item(),
                            "probability": prob.item(),
                        }
                        self.predictions.append(d)



                if model_name not in model_groups:
                    model_groups[model_name] = []
                model_groups[model_name].append(pipe)

                if len(model_groups[model_name]) == self.k_folds:
                    print(
                        f"[{perf_counter() - fit_start_time:.4f}s] Completed training for model {model_name}"
                    )

                    self.trained_models_.append((model_name, model_groups[model_name]))
                    
                    del model_groups[model_name]

                    # get updated Xt
                    Xt = self.get_Xt()
                    prob_columns = [col for col in Xt.columns if model_name in col]
                    agg_probs = Xt.select(prob_columns)
                    oof_probas = agg_probs.to_numpy()
                    oof_pred_indices = np.argmax(oof_probas, axis=1)
                    oof_preds = self.classes_[oof_pred_indices]
                    oof_acc = accuracy_score(y, oof_preds)

                    print(
                        f"[{perf_counter() - fit_start_time:.4f}s] OOF acc for model {model_name}: {oof_acc}"
                    )

            Xt = self.get_Xt()
            rXt = ray.put(Xt.to_numpy())
            rXt_columns = ray.put(Xt.columns)

            for model_name in self.stacking_models:
                model_group = []

                # RAY ------------------
                tasks = []
                for fold_number, (train_idx, val_idx) in enumerate(current_splits):
                    use_Xt = model_name not in self.series_models
                    model_seed = self._get_feature_seed()
                    run_queue.put((fold_number, model_name, use_Xt, train_idx, val_idx, model_seed))
                for _ in range(self.n_jobs):
                    run_queue.put(None)

                for i in range(self.n_jobs):
                    tasks.append(train_predict.remote(i, run_queue, result_queue, rX, rXt, ry, rXt_columns))

                for _ in range(len(current_splits)):
                    worker_id, train_idx, val_idx, proba, pickle_pipe, train_dur, model_name, fold_number = result_queue.get()
                    pipe = pickle.loads(pickle_pipe)

                    print(
                        f"[{perf_counter() - fit_start_time:.4f}s] Trained {model_name} on {worker_id} in {train_dur:.4f}s for f-{fold_number}/r-{repetition}"
                    )       

                    level = 0 if model_name in self.feature_models + self.series_models else 1

                    for idx, p in zip(val_idx, proba):
                        for scls, prob in zip(pipe.classes_, p):
                            d = {
                                "index": idx,
                                "model": model_name,
                                "repetition": repetition,
                                "level": level,
                                "class": scls.item(),
                                "probability": prob.item(),
                            }
                            self.predictions.append(d)
                    model_group.append(pipe)

                self.trained_models_.append((model_name, model_group))
                ray.get(tasks)
                # RAY ------------------ 

                # get updated Xt
                Xt = self.get_Xt()
                prob_columns = [col for col in Xt.columns if model_name in col]
                agg_probs = Xt.select(prob_columns)
                oof_probas = agg_probs.to_numpy()
                oof_pred_indices = np.argmax(oof_probas, axis=1)
                oof_preds = self.classes_[oof_pred_indices]
                oof_acc = accuracy_score(y, oof_preds)

                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] OOF acc for model {model_name}: {oof_acc}"
                )

            print(f"[{perf_counter() - fit_start_time:.4f}s] Completed repetition {repetition}")
        print(f"[{perf_counter() - fit_start_time:.4f}s] Completed all repetitions")
        self.best_model = "probability-ridgecv"
        ray.shutdown()

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

    def add_probabilities(self, probas, classes, model_name, level):
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
        print(f"[{perf_counter() - predict_start_time:.4f}s] Starting prediction")

        self._init_ray()

        return_dict = {}
        Xt = self.compute_features(X)
        print(f"[{perf_counter() - predict_start_time:.4f}s] Computed features for prediction")

        # Put data into Ray object store
        rX = ray.put(X)
        rXt = ray.put(Xt.to_numpy())
        rXt_columns = ray.put(Xt.columns)
        print(f"[{perf_counter() - predict_start_time:.4f}s] Data added to ray storage")

        predictions = []
        run_queue = Queue()
        result_queue = Queue()
        tasks = []

        # Step 1: Predict with all first-level models (feature_models + series_models) in parallel
        first_level_models = [(model_name, model_group) for model_name, model_group in self.trained_models_
                              if model_name in self.feature_models + self.series_models]

        model_count = 0
        for model_name, model_group in first_level_models:
            use_series = model_name in self.series_models
            for model in model_group:
                run_queue.put((model_name, pickle.dumps(model), use_series))
                model_count += 1

        for _ in range(self.n_jobs):
            run_queue.put(None)

        print(f"[{perf_counter() - predict_start_time:.4f}s] Starting RAY prediction with {self.n_jobs} workers for {model_count} first-level models")

        for i in range(self.n_jobs):
            tasks.append(predict_worker.remote(i, run_queue, result_queue, rX, rXt, rXt_columns))

        # Collect all first-level predictions
        for _ in range(model_count):
            worker_id, proba, classes_, predict_dur, model_name = result_queue.get()
            print(
                f"[{perf_counter() - predict_start_time:.4f}s] Predicted {model_name} on worker-{worker_id} in {predict_dur:.4f}s"
            )

            level = 0
            pred_list = self.add_probabilities(proba, classes_, model_name, level)
            predictions.extend(pred_list)

        print(f"[{perf_counter() - predict_start_time:.4f}s] Completed all first-level model predictions")
        ray.get(tasks)
        
        # Step 2: Now handle stacking models (second-level)
        # Update Xt with first-level predictions and put back in Ray
        Xt = self.combine_features_and_predictions(Xt, predictions)
        rXt = ray.put(Xt.to_numpy())
        rXt_columns = ray.put(Xt.columns)

        stacking_models = [(model_name, model_group) for model_name, model_group in self.trained_models_
                           if model_name in self.stacking_models]

        run_queue = Queue()
        result_queue = Queue()
        tasks = []

        model_count = 0
        for model_name, model_group in stacking_models:
            use_series = model_name in self.series_models
            for model in model_group:
                run_queue.put((model_name, pickle.dumps(model), use_series))
                model_count += 1

        for _ in range(self.n_jobs):
            run_queue.put(None)

        print(f"[{perf_counter() - predict_start_time:.4f}s] Starting RAY prediction with {self.n_jobs} workers for {model_count} stacking models")

        for i in range(self.n_jobs):
            tasks.append(predict_worker.remote(i, run_queue, result_queue, rX, rXt, rXt_columns))

        # Collect all stacking predictions
        for _ in range(model_count):
            worker_id, proba, classes_, predict_dur, model_name = result_queue.get()
            print(
                f"[{perf_counter() - predict_start_time:.4f}s] Predicted {model_name} on worker-{worker_id} in {predict_dur:.4f}s"
            )

            level = 1
            pred_list = self.add_probabilities(proba, classes_, model_name, level)
            predictions.extend(pred_list)

        print(f"[{perf_counter() - predict_start_time:.4f}s] Completed all stacking model predictions")

        df = self.combine_features_and_predictions(Xt, predictions)
        for model_name, _ in self.trained_models_:
            prob_columns = [col for col in df.columns if model_name in col]
            agg_probs = df.select(prob_columns)
            return_dict[model_name] = agg_probs.to_numpy()

        ray.shutdown()

        return return_dict

    def _predict_proba(self, X):
        return self.predict_proba_per_model(X)[self.best_model]

    def _predict(self, X):
        probas = self._predict_proba(X)
        predicted_indices = np.argmax(probas, axis=1)
        return self.classes_[predicted_indices]

    def add_features(self, feature_type: str, X: np.ndarray):
        transform = self.get_next_feature_transformer(feature_type=feature_type)
        transform.fit(X)
        X_t = transform.transform(X)
        params = transform.get_params()
        if "n_jobs" in params:
            del params["n_jobs"]
        param_strs = [f"{k}={v}" for k, v in params.items()]
        param_str = ";".join(param_strs)
        transform_id = f"{transform.__class__.__name__.lower()};{param_str}"
        schema = ["feature|" + transform_id + ";index=" + str(i) for i in range(X_t.shape[1])]
        feature_df = pl.DataFrame(X_t, schema=schema)
        self.feature_transformers.append(transform)
        if self.features is None:
            self.features = feature_df
        else:
            self.features = pl.concat([self.features, feature_df], how="horizontal")

    def compute_features(self, X: np.ndarray):
        feature_dfs = []
        for transform in self.feature_transformers:
            X_t = transform.transform(X)
            params = transform.get_params()
            if "n_jobs" in params:
                del params["n_jobs"]
            param_strs = [f"{k}={v}" for k, v in params.items()]
            param_str = ";".join(param_strs)
            transform_id = f"{transform.__class__.__name__.lower()};{param_str}"
            schema = ["feature|" + transform_id + ";index=" + str(i) for i in range(X_t.shape[1])]
            feature_df = pl.DataFrame(X_t, schema=schema)
            feature_dfs.append(feature_df)
        return pl.concat(feature_dfs, how="horizontal")

    def get_Xt(self) -> pl.DataFrame:
        return self.combine_features_and_predictions(self.features, self.predictions)


class FastStackerV5(BaseClassifier):
    """Like FastStackerV4 but only trains stacking model once after all repetitions complete."""
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, time_limit_in_seconds=None, n_jobs=1,
                 isolated_ray=True, ray_port=None):
        super().__init__()
        self.n_repetitions = n_repetitions
        self.k_folds = k_folds
        self.random_state = random_state
        self.cv_splits = None
        self.n_jobs = n_jobs
        self.isolated_ray = isolated_ray
        self.ray_port = ray_port

        self.feature_seed = np.random.default_rng(random_state)
        self.feature_transformers = []
        self.features = None
        self.predictions = None
        self.trained_models_ = None
        self.time_limit_in_seconds = time_limit_in_seconds

    def _get_available_port(self):
        """Find a random available port in range 60000-65535."""
        import socket
        import random
        for _ in range(100):
            port = random.randint(60000, 65535)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('', port))
                    return port
                except OSError:
                    continue
        raise RuntimeError("Could not find an available port in range 60000-65535")

    def _init_ray(self):
        """Initialize Ray, optionally as an isolated instance with unique temp dir."""
        if self.isolated_ray:
            if self.ray_port is None:
                self.ray_port = self._get_available_port()
            ray_tmp = os.path.expanduser(f"~/ray/p_{self.ray_port}")
            ray_spill = os.path.expanduser(f"~/ray/s_{self.ray_port}")
            os.makedirs(ray_tmp, exist_ok=True)
            os.makedirs(ray_spill, exist_ok=True)
            print(f"Starting isolated Ray on port {self.ray_port} with temp_dir={ray_tmp}")
            ray.init(
                num_cpus=self.n_jobs,
                ignore_reinit_error=False,
                include_dashboard=False,
                _temp_dir=ray_tmp,
                _system_config={
                    "object_spilling_config": json.dumps({
                        "type": "filesystem",
                        "params": {"directory_path": ray_spill}
                    }),
                    "memory_monitor_refresh_ms": 0,
                },
            )
        else:
            print("Starting shared Ray instance")
            ray.init(
                num_cpus=self.n_jobs,
                ignore_reinit_error=True,
            )

    def _get_feature_seed(self):
        return int(self.feature_seed.integers(0, 2**31 - 1, dtype=np.int32))

    def get_next_feature_transformer(self, feature_type: str):
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

    def _fit(self, X, y):
        fit_start_time = perf_counter()

        self.predictions = []
        self.trained_models_ = []

        self.feature_models = ["multirockethydra-ridgecv", "quant-etc", "rdst-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []
        self.stacking_models = ["probability-ridgecv"]

        if self.cv_splits is None:
            self.cv_splits = []
        print(f"[{perf_counter() - fit_start_time:.4f}s] Starting fitting")

        quant_start_time = perf_counter()
        self.add_features(feature_type="quant", X=X)
        quant_durration = perf_counter() - quant_start_time
        print(
            f"[{perf_counter() - fit_start_time:.4f}s] Computed QUANT features in {quant_durration:.4f}s"
        )

        self._init_ray()

        # Accumulate all splits across repetitions for stacking
        all_splits = []

        for repetition in range(self.n_repetitions):
            print(f"[{perf_counter() - fit_start_time:.4f}s] Starting repetition {repetition}")

            multirocket_start_time = perf_counter()
            self.add_features(feature_type="multirocket", X=X)
            multirocket_durration = perf_counter() - multirocket_start_time
            print(
                f"[{perf_counter() - fit_start_time:.4f}s] Computed MultiRocket features in {multirocket_durration:.4f}s"
            )

            hydra_start_time = perf_counter()
            self.add_features(feature_type="hydra", X=X)
            hydra_durration = perf_counter() - hydra_start_time
            print(
                f"[{perf_counter() - fit_start_time:.4f}s] Computed Hydra features in {hydra_durration:.4f}s"
            )

            rdst_start_time = perf_counter()
            self.add_features(feature_type="rdst", X=X)
            rdst_durration = perf_counter() - rdst_start_time
            print(
                f"[{perf_counter() - fit_start_time:.4f}s] Computed RDST features in {rdst_durration:.4f}s"
            )

            current_splits = generate_folds(
                X, y, n_splits=self.k_folds, n_repetitions=1, random_state=self._get_feature_seed()
            )
            all_splits.extend(current_splits)

            rX = ray.put(X)
            ry = ray.put(y)
            run_queue = Queue()
            result_queue = Queue()

            Xt = self.get_Xt()
            rXt = ray.put(Xt.to_numpy())
            rXt_columns = ray.put(Xt.columns)
            tasks = []

            print(f"[{perf_counter() - fit_start_time:.4f}s] Data added to ray storage")

            model_count = 0
            for model_name in self.feature_models + self.series_models:
                for fold_number, (train_idx, val_idx) in enumerate(current_splits):
                    use_Xt = model_name not in self.series_models
                    model_seed = self._get_feature_seed()
                    run_queue.put((fold_number, model_name, use_Xt, train_idx, val_idx, model_seed))
                    model_count += 1

            for _ in range(self.n_jobs):
                run_queue.put(None)

            print(f"[{perf_counter() - fit_start_time:.4f}s] Starting RAY training with {self.n_jobs} workers for {model_count} models")

            for i in range(self.n_jobs):
                tasks.append(train_predict.remote(i, run_queue, result_queue, rX, rXt, ry, rXt_columns))

            model_groups = {}
            for _ in range(model_count):
                worker_id, train_idx, val_idx, proba, pickle_pipe, train_dur, model_name, fold_number = result_queue.get()
                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] Trained {model_name} on {worker_id} in {train_dur:.4f}s for f-{fold_number}/r-{repetition}"
                )
                pipe = pickle.loads(pickle_pipe)

                level = 0 if model_name in self.feature_models + self.series_models else 1
                for idx, p in zip(val_idx, proba):
                    for scls, prob in zip(pipe.classes_, p):
                        d = {
                            "index": idx,
                            "model": model_name,
                            "repetition": repetition,
                            "level": level,
                            "class": scls.item(),
                            "probability": prob.item(),
                        }
                        self.predictions.append(d)

                if model_name not in model_groups:
                    model_groups[model_name] = []
                model_groups[model_name].append(pipe)

                if len(model_groups[model_name]) == self.k_folds:
                    print(
                        f"[{perf_counter() - fit_start_time:.4f}s] Completed training for model {model_name}"
                    )

                    self.trained_models_.append((model_name, model_groups[model_name]))

                    del model_groups[model_name]

                    Xt = self.get_Xt()
                    prob_columns = [col for col in Xt.columns if model_name in col]
                    agg_probs = Xt.select(prob_columns)
                    oof_probas = agg_probs.to_numpy()
                    oof_pred_indices = np.argmax(oof_probas, axis=1)
                    oof_preds = self.classes_[oof_pred_indices]
                    oof_acc = accuracy_score(y, oof_preds)

                    print(
                        f"[{perf_counter() - fit_start_time:.4f}s] OOF acc for model {model_name}: {oof_acc}"
                    )

            ray.get(tasks)
            print(f"[{perf_counter() - fit_start_time:.4f}s] Completed repetition {repetition}")

        # Train stacking models only once after all repetitions
        print(f"[{perf_counter() - fit_start_time:.4f}s] Starting stacking model training (single pass)")

        Xt = self.get_Xt()
        rX = ray.put(X)
        ry = ray.put(y)
        rXt = ray.put(Xt.to_numpy())
        rXt_columns = ray.put(Xt.columns)
        run_queue = Queue()
        result_queue = Queue()

        for model_name in self.stacking_models:
            model_group = []

            tasks = []
            for fold_number, (train_idx, val_idx) in enumerate(all_splits):
                use_Xt = model_name not in self.series_models
                model_seed = self._get_feature_seed()
                run_queue.put((fold_number, model_name, use_Xt, train_idx, val_idx, model_seed))
            for _ in range(self.n_jobs):
                run_queue.put(None)

            for i in range(self.n_jobs):
                tasks.append(train_predict.remote(i, run_queue, result_queue, rX, rXt, ry, rXt_columns))

            for _ in range(len(all_splits)):
                worker_id, train_idx, val_idx, proba, pickle_pipe, train_dur, model_name, fold_number = result_queue.get()
                pipe = pickle.loads(pickle_pipe)

                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] Trained {model_name} on {worker_id} in {train_dur:.4f}s for f-{fold_number}"
                )

                level = 1

                for idx, p in zip(val_idx, proba):
                    for scls, prob in zip(pipe.classes_, p):
                        d = {
                            "index": idx,
                            "model": model_name,
                            "repetition": 0,
                            "level": level,
                            "class": scls.item(),
                            "probability": prob.item(),
                        }
                        self.predictions.append(d)
                model_group.append(pipe)

            self.trained_models_.append((model_name, model_group))
            ray.get(tasks)

            Xt = self.get_Xt()
            prob_columns = [col for col in Xt.columns if model_name in col]
            agg_probs = Xt.select(prob_columns)
            oof_probas = agg_probs.to_numpy()
            oof_pred_indices = np.argmax(oof_probas, axis=1)
            oof_preds = self.classes_[oof_pred_indices]
            oof_acc = accuracy_score(y, oof_preds)

            print(
                f"[{perf_counter() - fit_start_time:.4f}s] OOF acc for model {model_name}: {oof_acc}"
            )

        print(f"[{perf_counter() - fit_start_time:.4f}s] Completed all repetitions and stacking")
        self.best_model = "probability-ridgecv"
        ray.shutdown()

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

    def add_probabilities(self, probas, classes, model_name, level):
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
        print(f"[{perf_counter() - predict_start_time:.4f}s] Starting prediction")

        self._init_ray()

        return_dict = {}
        Xt = self.compute_features(X)
        print(f"[{perf_counter() - predict_start_time:.4f}s] Computed features for prediction")

        rX = ray.put(X)
        rXt = ray.put(Xt.to_numpy())
        rXt_columns = ray.put(Xt.columns)
        print(f"[{perf_counter() - predict_start_time:.4f}s] Data added to ray storage")

        predictions = []
        run_queue = Queue()
        result_queue = Queue()
        tasks = []

        first_level_models = [(model_name, model_group) for model_name, model_group in self.trained_models_
                              if model_name in self.feature_models + self.series_models]

        model_count = 0
        for model_name, model_group in first_level_models:
            use_series = model_name in self.series_models
            for model in model_group:
                run_queue.put((model_name, pickle.dumps(model), use_series))
                model_count += 1

        for _ in range(self.n_jobs):
            run_queue.put(None)

        print(f"[{perf_counter() - predict_start_time:.4f}s] Starting RAY prediction with {self.n_jobs} workers for {model_count} first-level models")

        for i in range(self.n_jobs):
            tasks.append(predict_worker.remote(i, run_queue, result_queue, rX, rXt, rXt_columns))

        for _ in range(model_count):
            worker_id, proba, classes_, predict_dur, model_name = result_queue.get()
            print(
                f"[{perf_counter() - predict_start_time:.4f}s] Predicted {model_name} on worker-{worker_id} in {predict_dur:.4f}s"
            )

            level = 0
            pred_list = self.add_probabilities(proba, classes_, model_name, level)
            predictions.extend(pred_list)

        print(f"[{perf_counter() - predict_start_time:.4f}s] Completed all first-level model predictions")
        ray.get(tasks)

        Xt = self.combine_features_and_predictions(Xt, predictions)
        rXt = ray.put(Xt.to_numpy())
        rXt_columns = ray.put(Xt.columns)

        stacking_models = [(model_name, model_group) for model_name, model_group in self.trained_models_
                           if model_name in self.stacking_models]

        run_queue = Queue()
        result_queue = Queue()
        tasks = []

        model_count = 0
        for model_name, model_group in stacking_models:
            use_series = model_name in self.series_models
            for model in model_group:
                run_queue.put((model_name, pickle.dumps(model), use_series))
                model_count += 1

        for _ in range(self.n_jobs):
            run_queue.put(None)

        print(f"[{perf_counter() - predict_start_time:.4f}s] Starting RAY prediction with {self.n_jobs} workers for {model_count} stacking models")

        for i in range(self.n_jobs):
            tasks.append(predict_worker.remote(i, run_queue, result_queue, rX, rXt, rXt_columns))

        for _ in range(model_count):
            worker_id, proba, classes_, predict_dur, model_name = result_queue.get()
            print(
                f"[{perf_counter() - predict_start_time:.4f}s] Predicted {model_name} on worker-{worker_id} in {predict_dur:.4f}s"
            )

            level = 1
            pred_list = self.add_probabilities(proba, classes_, model_name, level)
            predictions.extend(pred_list)

        print(f"[{perf_counter() - predict_start_time:.4f}s] Completed all stacking model predictions")

        df = self.combine_features_and_predictions(Xt, predictions)
        for model_name, _ in self.trained_models_:
            prob_columns = [col for col in df.columns if model_name in col]
            agg_probs = df.select(prob_columns)
            return_dict[model_name] = agg_probs.to_numpy()

        ray.shutdown()

        return return_dict

    def _predict_proba(self, X):
        return self.predict_proba_per_model(X)[self.best_model]

    def _predict(self, X):
        probas = self._predict_proba(X)
        predicted_indices = np.argmax(probas, axis=1)
        return self.classes_[predicted_indices]

    def add_features(self, feature_type: str, X: np.ndarray):
        transform = self.get_next_feature_transformer(feature_type=feature_type)
        transform.fit(X)
        X_t = transform.transform(X)
        params = transform.get_params()
        if "n_jobs" in params:
            del params["n_jobs"]
        param_strs = [f"{k}={v}" for k, v in params.items()]
        param_str = ";".join(param_strs)
        transform_id = f"{transform.__class__.__name__.lower()};{param_str}"
        schema = ["feature|" + transform_id + ";index=" + str(i) for i in range(X_t.shape[1])]
        feature_df = pl.DataFrame(X_t, schema=schema)
        self.feature_transformers.append(transform)
        if self.features is None:
            self.features = feature_df
        else:
            self.features = pl.concat([self.features, feature_df], how="horizontal")

    def compute_features(self, X: np.ndarray):
        feature_dfs = []
        for transform in self.feature_transformers:
            X_t = transform.transform(X)
            params = transform.get_params()
            if "n_jobs" in params:
                del params["n_jobs"]
            param_strs = [f"{k}={v}" for k, v in params.items()]
            param_str = ";".join(param_strs)
            transform_id = f"{transform.__class__.__name__.lower()};{param_str}"
            schema = ["feature|" + transform_id + ";index=" + str(i) for i in range(X_t.shape[1])]
            feature_df = pl.DataFrame(X_t, schema=schema)
            feature_dfs.append(feature_df)
        return pl.concat(feature_dfs, how="horizontal")

    def get_Xt(self) -> pl.DataFrame:
        return self.combine_features_and_predictions(self.features, self.predictions)


# Global worker data - loaded once per worker process via initializer
# Each worker subprocess has its own copy, freed when executor shuts down
_loky_worker_data = None


def _init_worker_train(X_path, Xt_path, Xt_columns_path, y_path):
    """Initialize worker by loading data from mmap files. Called once per worker."""
    global _loky_worker_data
    import os
    pid = os.getpid()
    print(f"[Worker {pid}] Train init starting...", flush=True)
    print(f"[Worker {pid}] Loading X...", flush=True)
    X = np.load(X_path, mmap_mode='r')
    print(f"[Worker {pid}] Loading Xt...", flush=True)
    Xt_np = np.load(Xt_path, mmap_mode='r')
    print(f"[Worker {pid}] Loading Xt_columns...", flush=True)
    with open(Xt_columns_path, 'rb') as f:
        Xt_columns = pickle.load(f)
    print(f"[Worker {pid}] Creating DataFrame ({len(Xt_columns)} cols)...", flush=True)
    Xt = pl.DataFrame(Xt_np, schema=Xt_columns)
    print(f"[Worker {pid}] Loading y...", flush=True)
    y = np.load(y_path, mmap_mode='r')
    _loky_worker_data = (X, Xt, y)
    print(f"[Worker {pid}] Train init done. X={X.shape}, Xt={Xt.shape}, y={y.shape}", flush=True)


def _init_worker_predict(X_path, Xt_path, Xt_columns_path):
    """Initialize worker for prediction by loading data from mmap files."""
    global _loky_worker_data
    import os
    pid = os.getpid()
    print(f"[Worker {pid}] Predict init starting...", flush=True)
    print(f"[Worker {pid}] Loading X...", flush=True)
    X = np.load(X_path, mmap_mode='r')
    print(f"[Worker {pid}] Loading Xt...", flush=True)
    Xt_np = np.load(Xt_path, mmap_mode='r')
    print(f"[Worker {pid}] Loading Xt_columns...", flush=True)
    with open(Xt_columns_path, 'rb') as f:
        Xt_columns = pickle.load(f)
    print(f"[Worker {pid}] Creating DataFrame ({len(Xt_columns)} cols)...", flush=True)
    Xt = pl.DataFrame(Xt_np, schema=Xt_columns)
    _loky_worker_data = (X, Xt, None)
    print(f"[Worker {pid}] Predict init done. X={X.shape}, Xt={Xt.shape}", flush=True)


def _train_one_model(fold_number, model_name, use_Xt, train_idx, val_idx, model_seed):
    """One-shot training function - uses pre-loaded data from initializer."""
    global _loky_worker_data
    import os
    pid = os.getpid()
    print(f"[Worker {pid}] Task start: {model_name} fold {fold_number}", flush=True)
    X, Xt, y = _loky_worker_data

    print(f"[Worker {pid}] Getting model {model_name}...", flush=True)
    pipe = get_model(model_name, seed=model_seed)
    start_train = perf_counter()

    print(f"[Worker {pid}] Fitting {model_name}...", flush=True)
    if use_Xt:
        pipe.fit(Xt[train_idx], y[train_idx])
        print(f"[Worker {pid}] Predicting {model_name}...", flush=True)
        proba = pipe.predict_proba(Xt[val_idx])
    else:
        pipe.fit(X[train_idx], y[train_idx])
        print(f"[Worker {pid}] Predicting {model_name}...", flush=True)
        proba = pipe.predict_proba(X[val_idx])

    train_dur = perf_counter() - start_train
    print(f"[Worker {pid}] Task done: {model_name} fold {fold_number} in {train_dur:.2f}s", flush=True)
    return (train_idx, val_idx, proba, pickle.dumps(pipe), train_dur, model_name, fold_number)


def _predict_one_model(model_name, pickle_pipe, use_series):
    """One-shot prediction function - uses pre-loaded data from initializer."""
    global _loky_worker_data
    import os
    pid = os.getpid()
    print(f"[Worker {pid}] Predict task start: {model_name}", flush=True)
    X, Xt, _ = _loky_worker_data

    print(f"[Worker {pid}] Unpickling model...", flush=True)
    pipe = pickle.loads(pickle_pipe)
    start_predict = perf_counter()

    print(f"[Worker {pid}] Running predict_proba...", flush=True)
    if use_series:
        proba = pipe.predict_proba(X)
    else:
        proba = pipe.predict_proba(Xt)

    predict_dur = perf_counter() - start_predict
    print(f"[Worker {pid}] Predict task done: {model_name} in {predict_dur:.2f}s", flush=True)
    return (proba, pipe.classes_, predict_dur, model_name)


class LokyStackerV5(BaseClassifier):
    """Like FastStackerV5 but uses loky instead of Ray for parallelization."""
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, time_limit_in_seconds=None, n_jobs=1):
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

        self._tmpdir = None
        self.fallback_model = None

    def _save_mmap_data(self, X, Xt, y):
        """Save data to mmap files for workers. Xt is saved as numpy array + columns."""
        X_path = f"{self._tmpdir}/X.npy"
        Xt_path = f"{self._tmpdir}/Xt.npy"
        Xt_columns_path = f"{self._tmpdir}/Xt_columns.pkl"
        y_path = f"{self._tmpdir}/y.npy"
        np.save(X_path, X)
        np.save(Xt_path, Xt.to_numpy())
        with open(Xt_columns_path, 'wb') as f:
            pickle.dump(Xt.columns, f)
        np.save(y_path, y)
        return X_path, Xt_path, Xt_columns_path, y_path

    def _get_feature_seed(self):
        return int(self.feature_seed.integers(0, 2**31 - 1, dtype=np.int32))

    def get_next_feature_transformer(self, feature_type: str, n_jobs: int = 1):
        seed = self._get_feature_seed()
        match feature_type:
            case "multirocket":
                return MultiRocket(n_jobs=n_jobs, random_state=seed)
            case "rdst":
                return RandomDilatedShapeletTransform(n_jobs=n_jobs, random_state=seed)
            case "quant":
                return QUANTTransformer()
            case "hydra":
                return HydraTransformer(n_jobs=n_jobs, random_state=seed)
            case _:
                raise ValueError(f"Unknown feature transformer type: {feature_type}")

    def _fit(self, X, y):
        import shutil
        fit_start_time = perf_counter()

        self.predictions = []
        self.trained_models_ = []

        self.feature_models = ["multirockethydra-ridgecv", "quant-etc", "rdst-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []
        self.stacking_models = ["probability-ridgecv"]

        if self.cv_splits is None:
            self.cv_splits = []
        print(f"[{perf_counter() - fit_start_time:.4f}s] Starting fitting")

        # Check if each class has at least 2 instances for fold training
        unique, counts = np.unique(y, return_counts=True)
        if np.any(counts < 2):
            print(f"[{perf_counter() - fit_start_time:.4f}s] Some classes have fewer than 2 instances, fold training not possible")
            print(f"[{perf_counter() - fit_start_time:.4f}s] Falling back to MultiRocketHydraClassifier")
            self.fallback_model = MultiRocketHydraClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            self.fallback_model.fit(X, y)
            print(f"[{perf_counter() - fit_start_time:.4f}s] Fallback model trained successfully")
            return

        quant_start_time = perf_counter()
        self.add_features(feature_type="quant", X=X)
        quant_durration = perf_counter() - quant_start_time
        print(
            f"[{perf_counter() - fit_start_time:.4f}s] Computed QUANT features in {quant_durration:.4f}s"
        )

        # Create temp directory for mmap files
        self._tmpdir = tempfile.mkdtemp(prefix="loky_stacker_")
        print(f"Starting executor with {self.n_jobs} workers, tmpdir={self._tmpdir}")

        # Accumulate all splits across repetitions for stacking
        all_splits = []

        try:
            for repetition in range(self.n_repetitions):
                print(f"[{perf_counter() - fit_start_time:.4f}s] Starting repetition {repetition}")

                multirocket_start_time = perf_counter()
                self.add_features(feature_type="multirocket", X=X)
                multirocket_durration = perf_counter() - multirocket_start_time
                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] Computed MultiRocket features in {multirocket_durration:.4f}s"
                )

                hydra_start_time = perf_counter()
                self.add_features(feature_type="hydra", X=X)
                hydra_durration = perf_counter() - hydra_start_time
                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] Computed Hydra features in {hydra_durration:.4f}s"
                )

                rdst_start_time = perf_counter()
                self.add_features(feature_type="rdst", X=X)
                rdst_durration = perf_counter() - rdst_start_time
                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] Computed RDST features in {rdst_durration:.4f}s"
                )

                current_splits = generate_folds(
                    X, y, n_splits=self.k_folds, n_repetitions=1, random_state=self._get_feature_seed()
                )
                all_splits.extend(current_splits)

                Xt = self.get_Xt()
                X_path, Xt_path, Xt_columns_path, y_path = self._save_mmap_data(X, Xt, y)

                print(f"[{perf_counter() - fit_start_time:.4f}s] Data saved to mmap files")

                # Build list of tasks for this repetition
                tasks = []
                for model_name in self.feature_models + self.series_models:
                    for fold_number, (train_idx, val_idx) in enumerate(current_splits):
                        use_Xt = model_name not in self.series_models
                        model_seed = self._get_feature_seed()
                        tasks.append((fold_number, model_name, use_Xt, train_idx, val_idx, model_seed))

                n_workers = min(self.n_jobs, len(tasks))
                print(f"[{perf_counter() - fit_start_time:.4f}s] Starting training with {n_workers} workers for {len(tasks)} models")

                # Use ProcessPoolExecutor with initializer - workers load data once
                # Use spawn to avoid fork+threading deadlocks
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=_init_worker_train,
                    initargs=(X_path, Xt_path, Xt_columns_path, y_path)
                ) as executor:
                    futures = {
                        executor.submit(_train_one_model, *task): task
                        for task in tasks
                    }

                    model_groups = {}
                    for future in as_completed(futures):
                        task = futures[future]
                        fold_number, model_name, _, _, _, _ = task
                        try:
                            result = future.result()
                        except Exception as e:
                            raise RuntimeError(f"Worker failed during training {model_name} fold {fold_number}: {e}")

                        train_idx, val_idx, proba, pickle_pipe, train_dur, model_name, fold_number = result
                        print(
                            f"[{perf_counter() - fit_start_time:.4f}s] Trained {model_name} in {train_dur:.4f}s for f-{fold_number}/r-{repetition}"
                        )
                        pipe = pickle.loads(pickle_pipe)

                        level = 0 if model_name in self.feature_models + self.series_models else 1
                        for idx, p in zip(val_idx, proba):
                            for scls, prob in zip(pipe.classes_, p):
                                d = {
                                    "index": idx,
                                    "model": model_name,
                                    "repetition": repetition,
                                    "level": level,
                                    "class": scls.item(),
                                    "probability": prob.item(),
                                }
                                self.predictions.append(d)

                        if model_name not in model_groups:
                            model_groups[model_name] = []
                        model_groups[model_name].append(pipe)

                        if len(model_groups[model_name]) == self.k_folds:
                            print(
                                f"[{perf_counter() - fit_start_time:.4f}s] Completed training for model {model_name}"
                            )

                            self.trained_models_.append((model_name, model_groups[model_name]))
                            del model_groups[model_name]

                            Xt = self.get_Xt()
                            prob_columns = [col for col in Xt.columns if model_name in col]
                            agg_probs = Xt.select(prob_columns)
                            oof_probas = agg_probs.to_numpy()
                            oof_pred_indices = np.argmax(oof_probas, axis=1)
                            oof_preds = self.classes_[oof_pred_indices]
                            oof_acc = accuracy_score(y, oof_preds)

                            print(
                                f"[{perf_counter() - fit_start_time:.4f}s] OOF acc for model {model_name}: {oof_acc}"
                            )

                print(f"[{perf_counter() - fit_start_time:.4f}s] Completed repetition {repetition}")

            # Train stacking models only once after all repetitions
            print(f"[{perf_counter() - fit_start_time:.4f}s] Starting stacking model training (single pass)")

            Xt = self.get_Xt()

            # Check for NaN values in probability columns (can happen when some folds don't have all classes)
            prob_columns = [col for col in Xt.columns if col.startswith("probability|")]
            has_nan = Xt.select(prob_columns).null_count().to_numpy().sum() > 0
            if has_nan:
                print(f"[{perf_counter() - fit_start_time:.4f}s] NaN values detected in probability columns, skipping stacking")
                print(f"[{perf_counter() - fit_start_time:.4f}s] Falling back to MultiRocketHydraClassifier")
                self.fallback_model = MultiRocketHydraClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
                self.fallback_model.fit(X, y)
                print(f"[{perf_counter() - fit_start_time:.4f}s] Fallback model trained successfully")
                return

            X_path, Xt_path, Xt_columns_path, y_path = self._save_mmap_data(X, Xt, y)

            for model_name in self.stacking_models:
                # Build tasks for stacking model
                tasks = []
                for fold_number, (train_idx, val_idx) in enumerate(all_splits):
                    use_Xt = model_name not in self.series_models
                    model_seed = self._get_feature_seed()
                    tasks.append((fold_number, model_name, use_Xt, train_idx, val_idx, model_seed))

                n_workers = min(self.n_jobs, len(tasks))
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=multiprocessing.get_context('spawn'),
                    initializer=_init_worker_train,
                    initargs=(X_path, Xt_path, Xt_columns_path, y_path)
                ) as executor:
                    futures = {
                        executor.submit(_train_one_model, *task): task
                        for task in tasks
                    }

                    model_group = []
                    for future in as_completed(futures):
                        task = futures[future]
                        fold_number, model_name_task, _, _, _, _ = task
                        try:
                            result = future.result()
                        except Exception as e:
                            raise RuntimeError(f"Worker failed during stacking training {model_name_task} fold {fold_number}: {e}")

                        train_idx, val_idx, proba, pickle_pipe, train_dur, model_name_result, fold_number = result
                        pipe = pickle.loads(pickle_pipe)

                        print(
                            f"[{perf_counter() - fit_start_time:.4f}s] Trained {model_name_result} in {train_dur:.4f}s for f-{fold_number}"
                        )

                        level = 1
                        for idx, p in zip(val_idx, proba):
                            for scls, prob in zip(pipe.classes_, p):
                                d = {
                                    "index": idx,
                                    "model": model_name_result,
                                    "repetition": 0,
                                    "level": level,
                                    "class": scls.item(),
                                    "probability": prob.item(),
                                }
                                self.predictions.append(d)
                        model_group.append(pipe)

                self.trained_models_.append((model_name, model_group))

                Xt = self.get_Xt()
                prob_columns = [col for col in Xt.columns if model_name in col]
                agg_probs = Xt.select(prob_columns)
                oof_probas = agg_probs.to_numpy()
                oof_pred_indices = np.argmax(oof_probas, axis=1)
                oof_preds = self.classes_[oof_pred_indices]
                oof_acc = accuracy_score(y, oof_preds)

                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] OOF acc for model {model_name}: {oof_acc}"
                )

            print(f"[{perf_counter() - fit_start_time:.4f}s] Completed all repetitions and stacking")
            self.best_model = "probability-ridgecv"

        finally:
            # Clean up temp directory
            if self._tmpdir and os.path.exists(self._tmpdir):
                shutil.rmtree(self._tmpdir)
                self._tmpdir = None
            print("Executor shutdown complete")

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

    def add_probabilities(self, probas, classes, model_name, level):
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
        import shutil
        predict_start_time = perf_counter()
        print(f"[{perf_counter() - predict_start_time:.4f}s] Starting prediction")

        # Create temp directory for mmap files
        self._tmpdir = tempfile.mkdtemp(prefix="loky_stacker_pred_")
        print(f"Starting executor with {self.n_jobs} workers, tmpdir={self._tmpdir}")

        try:
            return_dict = {}
            Xt = self.compute_features(X)
            print(f"[{perf_counter() - predict_start_time:.4f}s] Computed features for prediction")

            X_path, Xt_path, Xt_columns_path, _ = self._save_mmap_data(X, Xt, np.array([]))
            print(f"[{perf_counter() - predict_start_time:.4f}s] Data saved to mmap files")

            predictions = []

            first_level_models = [(model_name, model_group) for model_name, model_group in self.trained_models_
                                  if model_name in self.feature_models + self.series_models]

            # Build tasks for first-level models
            tasks = []
            for model_name, model_group in reversed(first_level_models):
                use_series = model_name in self.series_models
                for model in model_group:
                    tasks.append((model_name, pickle.dumps(model), use_series))

            n_workers = min(self.n_jobs, len(tasks))
            print(f"[{perf_counter() - predict_start_time:.4f}s] Starting prediction with {n_workers} workers for {len(tasks)} first-level models")

            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_init_worker_predict,
                initargs=(X_path, Xt_path, Xt_columns_path)
            ) as executor:
                futures = {
                    executor.submit(_predict_one_model, *task): task
                    for task in tasks
                }

                for future in as_completed(futures):
                    task = futures[future]
                    model_name_task = task[0]
                    try:
                        result = future.result()
                    except Exception as e:
                        raise RuntimeError(f"Worker failed during prediction {model_name_task}: {e}")

                    proba, classes_, predict_dur, model_name = result
                    print(
                        f"[{perf_counter() - predict_start_time:.4f}s] Predicted {model_name} in {predict_dur:.4f}s"
                    )

                    level = 0
                    pred_list = self.add_probabilities(proba, classes_, model_name, level)
                    predictions.extend(pred_list)

            print(f"[{perf_counter() - predict_start_time:.4f}s] Completed all first-level model predictions")

            # Clean up first executor's mmap files before creating new ones
            if self._tmpdir and os.path.exists(self._tmpdir):
                shutil.rmtree(self._tmpdir)
            self._tmpdir = tempfile.mkdtemp(prefix="loky_stacker_pred_stack_")

            Xt = self.combine_features_and_predictions(Xt, predictions)
            X_path, Xt_path, Xt_columns_path, _ = self._save_mmap_data(X, Xt, np.array([]))

            stacking_models = [(model_name, model_group) for model_name, model_group in self.trained_models_
                               if model_name in self.stacking_models]

            # Build tasks for stacking models
            tasks = []
            for model_name, model_group in reversed(stacking_models):
                use_series = model_name in self.series_models
                for model in model_group:
                    tasks.append((model_name, pickle.dumps(model), use_series))

            n_workers = min(self.n_jobs, len(tasks))
            print(f"[{perf_counter() - predict_start_time:.4f}s] Starting prediction with {n_workers} workers for {len(tasks)} stacking models")

            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=_init_worker_predict,
                initargs=(X_path, Xt_path, Xt_columns_path)
            ) as executor:
                futures = {
                    executor.submit(_predict_one_model, *task): task
                    for task in tasks
                }

                for future in as_completed(futures):
                    task = futures[future]
                    model_name_task = task[0]
                    try:
                        result = future.result()
                    except Exception as e:
                        raise RuntimeError(f"Worker failed during stacking prediction {model_name_task}: {e}")

                    proba, classes_, predict_dur, model_name = result
                    print(
                        f"[{perf_counter() - predict_start_time:.4f}s] Predicted {model_name} in {predict_dur:.4f}s"
                    )

                    level = 1
                    pred_list = self.add_probabilities(proba, classes_, model_name, level)
                    predictions.extend(pred_list)

            print(f"[{perf_counter() - predict_start_time:.4f}s] Completed all stacking model predictions")

            df = self.combine_features_and_predictions(Xt, predictions)
            for model_name, _ in self.trained_models_:
                prob_columns = [col for col in df.columns if model_name in col]
                agg_probs = df.select(prob_columns)
                return_dict[model_name] = agg_probs.to_numpy()

            return return_dict

        finally:
            # Clean up temp directory
            if self._tmpdir and os.path.exists(self._tmpdir):
                shutil.rmtree(self._tmpdir)
                self._tmpdir = None
            print("Executor shutdown complete")

    def _predict_proba(self, X):
        if self.fallback_model is not None:
            return self.fallback_model.predict_proba(X)
        return self.predict_proba_per_model(X)[self.best_model]

    def _predict(self, X):
        if self.fallback_model is not None:
            return self.fallback_model.predict(X)
        probas = self._predict_proba(X)
        predicted_indices = np.argmax(probas, axis=1)
        return self.classes_[predicted_indices]

    def add_features(self, feature_type: str, X: np.ndarray):
        transform = self.get_next_feature_transformer(feature_type=feature_type, n_jobs=self.n_jobs)
        transform.fit(X)
        X_t = transform.transform(X)
        params = transform.get_params()
        if "n_jobs" in params:
            del params["n_jobs"]
        param_strs = [f"{k}={v}" for k, v in params.items()]
        param_str = ";".join(param_strs)
        transform_id = f"{transform.__class__.__name__.lower()};{param_str}"
        schema = ["feature|" + transform_id + ";index=" + str(i) for i in range(X_t.shape[1])]
        feature_df = pl.DataFrame(X_t, schema=schema)
        self.feature_transformers.append(transform)
        if self.features is None:
            self.features = feature_df
        else:
            self.features = pl.concat([self.features, feature_df], how="horizontal")

    def compute_features(self, X: np.ndarray):
        feature_dfs = []
        for transform in self.feature_transformers:
            X_t = transform.transform(X)
            params = transform.get_params()
            if "n_jobs" in params:
                del params["n_jobs"]
            param_strs = [f"{k}={v}" for k, v in params.items()]
            param_str = ";".join(param_strs)
            transform_id = f"{transform.__class__.__name__.lower()};{param_str}"
            schema = ["feature|" + transform_id + ";index=" + str(i) for i in range(X_t.shape[1])]
            feature_df = pl.DataFrame(X_t, schema=schema)
            feature_dfs.append(feature_df)
        return pl.concat(feature_dfs, how="horizontal")

    def get_Xt(self) -> pl.DataFrame:
        return self.combine_features_and_predictions(self.features, self.predictions)


def generate_folds(X, y, n_splits=5, n_repetitions=5, random_state=0):
    all_folds = []
    for i in range(n_repetitions):
        folds = utils.get_folds(X, y, n_splits=n_splits, random_state=random_state + i)
        all_folds.extend(folds)
    return all_folds


class CrossValidationWrapper(BaseClassifier):
    def __init__(self, model, k_folds=10, n_repetitions=1, random_state=None):
        super().__init__()
        self.model = model
        self.trained_models_ = []
        self.fit_time_ = []
        self.fit_time_mean_ = None
        self.predict_time_ = []
        self.predict_time_mean_ = None
        self.cv_splits = None
        self.k_folds = k_folds
        self.n_repetitions = n_repetitions
        self.random_state = random_state

    def _fit(self, X, y):
        raise NotImplementedError()

    def _predict_proba(self, X):
        predictions = []
        for model in self.trained_models_:
            proba = model.predict_proba(X)
            predictions.append(proba)
        avg_proba = np.mean(predictions, axis=0)
        return avg_proba

    def _predict(self, X):
        probas = self._predict_proba(X)
        predicted_indices = np.argmax(probas, axis=1)
        return self.classes_[predicted_indices]

    def get_all_oof_proba(self):
        return pl.DataFrame(self.oof_proba).sort("index", maintain_order=True)

    def _fit_predict_proba(self, X, y):
        if self.cv_splits is None:
            self.cv_splits = generate_folds(
                X,
                y,
                n_splits=self.k_folds,
                n_repetitions=self.n_repetitions,
                random_state=self.random_state,
            )
        self.oof_proba = []
        for train_idx, val_idx in tqdm(self.cv_splits):
            model_clone = self.model.clone()
            # print(model_clone)
            # print('RS:', model_clone.random_state)
            X_train, y_train = X[train_idx], y[train_idx]
            X_valid, _ = X[val_idx], y[val_idx]
            model_clone.fit(X_train, y_train)
            self.trained_models_.append(model_clone)
            proba = model_clone.predict_proba(X_valid)
            # print(val_idx, train_idx)
            # print(proba)
            prob_columns = []
            for idx, p in zip(val_idx, proba):
                d = {
                    "index": idx,
                }
                for scls, prob in zip(model_clone.classes_, p):
                    k = f"proba_class_{scls}"
                    d[k] = prob.item()
                    if k not in prob_columns:
                        prob_columns.append(k)
                self.oof_proba.append(d)
        return (
            pl.DataFrame(self.oof_proba)
            .group_by("index")
            .mean()
            .sort("index")
            .select(prob_columns)
            .to_numpy()
        )

class FeatureCrossValidationWrapper(BaseClassifier):
    def __init__(self, features, model, k_folds=10, n_repetitions=1, random_state=None):
        super().__init__()
        self.features = features.clone()
        self.model = model
        self.trained_models_ = []
        self.fit_time_ = []
        self.fit_time_mean_ = None
        self.predict_time_ = []
        self.predict_time_mean_ = None
        self.cv_splits = None
        self.k_folds = k_folds
        self.n_repetitions = n_repetitions
        self.random_state = random_state

    def _fit(self, X, y):
        raise NotImplementedError()

    def _fit_predict_proba(self, X, y):
        if self.cv_splits is None:
            self.cv_splits = generate_folds(
                X,
                y,
                n_splits=self.k_folds,
                n_repetitions=self.n_repetitions,
                random_state=self.random_state,
            )
        self.oof_proba = []
        Xt = self.features.fit_transform(X)
        for train_idx, val_idx in tqdm(self.cv_splits):
            model_clone = clone(self.model)
            # print(model_clone)
            # print('RS:', model_clone.random_state)
            X_train, y_train = Xt[train_idx], y[train_idx]
            X_valid, _ = Xt[val_idx], y[val_idx]
            model_clone.fit(X_train, y_train)
            self.trained_models_.append(model_clone)
            proba = model_clone.predict_proba(X_valid)
            # print(val_idx, train_idx)
            # print(proba)
            # break
            prob_columns = []
            for idx, p in zip(val_idx, proba):
                d = {
                    "index": idx,
                }
                for scls, prob in zip(model_clone.classes_, p):
                    k = f"proba_class_{scls}"
                    d[k] = prob.item()
                    if k not in prob_columns:
                        prob_columns.append(k)
                self.oof_proba.append(d)
        return (
            pl.DataFrame(self.oof_proba)
            .group_by("index")
            .mean()
            .sort("index")
            .select(prob_columns)
            .to_numpy()
        )

    def _predict_proba(self, X):
        Xt = self.features.transform(X)
        predictions = []
        for model in self.trained_models_:
            proba = model.predict_proba(Xt)
            predictions.append(proba)
        avg_proba = np.mean(predictions, axis=0)
        return avg_proba

    def _predict(self, X):
        probas = self._predict_proba(X)
        predicted_indices = np.argmax(probas, axis=1)
        return self.classes_[predicted_indices]


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


from sklearn.base import BaseEstimator, TransformerMixin


class DualScaler(BaseEstimator, TransformerMixin):
    """
    Scales columns based on prefix:
    - 'hydra': hydra_scaler (default: SparseScaler)
    - 'multirocket': rocket_scaler (default: StandardScaler)
    - Other: raises ValueError
    """

    def __init__(self, hydra_scaler, rocket_scaler):
        self.hydra_scaler = hydra_scaler
        self.rocket_scaler = rocket_scaler

    def fit(self, X, y=None):
        # Initialize scalers
        self.hydra_scaler_ = self.hydra_scaler
        self.rocket_scaler_ = self.rocket_scaler

        # Separate columns by prefix
        self.hydra_cols_ = [col for col in X.columns if col.startswith("hydra")]
        self.multirocket_cols_ = [col for col in X.columns if col.startswith("multirocket")]

        # Check for invalid columns
        valid_cols = set(self.hydra_cols_ + self.multirocket_cols_)
        invalid_cols = [col for col in X.columns if col not in valid_cols]
        if invalid_cols:
            raise ValueError(
                f"Invalid column prefixes found: {invalid_cols}. Only 'hydra' and 'multirocket' prefixes are allowed."
            )

        # Fit scalers
        if self.hydra_cols_:
            self.hydra_scaler_.fit(X.select(self.hydra_cols_).to_numpy())
        if self.multirocket_cols_:
            self.rocket_scaler_.fit(X.select(self.multirocket_cols_).to_numpy())

        return self

    def transform(self, X):
        # Transform each group
        parts = []
        if self.hydra_cols_:
            parts.append(self.hydra_scaler_.transform(X.select(self.hydra_cols_).to_numpy()))
        if self.multirocket_cols_:
            parts.append(self.rocket_scaler_.transform(X.select(self.multirocket_cols_).to_numpy()))

        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        """Return feature names in output order."""
        return np.array(self.hydra_cols_ + self.multirocket_cols_)


from aeon.transformations.collection import BaseCollectionTransformer


class MultiRocketHydra(BaseCollectionTransformer):
    _tags = {
        "output_data_type": "Tabular",
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
        "X_inner_type": "numpy3D",
    }

    def __init__(
        self,
        n_jobs=1,
        random_state=None,
    ):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.multirocket_ = MultiRocket(random_state=self.random_state, n_jobs=self.n_jobs)
        self.hydra_ = HydraTransformer(random_state=self.random_state, n_jobs=self.n_jobs)
        self.n_multirocket_features_ = None
        self.n_hydra_features_ = None
        super().__init__()

    def _fit(self, X, y=None):
        self.hydra_.fit(X)
        self.multirocket_.fit(X)
        return self

    def _transform(self, X, y=None):
        X_hydra_t = self.hydra_.transform(X)
        X_multirocket_t = self.multirocket_.transform(X)
        Xt = np.concatenate((X_hydra_t, X_multirocket_t), axis=1)
        self.n_multirocket_features_ = X_multirocket_t.shape[1]
        self.n_hydra_features_ = X_hydra_t.shape[1]

        schema = [f"hydra_{x}" for x in range(self.n_hydra_features_)] + [
            f"multirocket_{x}" for x in range(self.n_multirocket_features_)
        ]
        return pl.DataFrame(Xt, schema=schema)


from time import perf_counter

import numpy as np
from aeon.transformations.collection.feature_based import Catch22
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


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

    def fit(self, X, y):
        with threadpool_limits(limits=1):
            return super().fit(X, y)
