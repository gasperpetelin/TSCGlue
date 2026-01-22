import json
import os
import tempfile

os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
import numpy as np
import polars as pl
import ray
from multiprocessing import Process, Queue as MPQueue, Manager
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor
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
                    })
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
                    })
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

class Stacker(BaseClassifier):
    def __init__(self, random_state=None, n_repetitions=1):
        super().__init__()
        self.n_repetitions = n_repetitions
        k_folds = 10
        self.random_state = random_state
        self.m1 = CrossValidationWrapper(
            MultiRocketHydraClassifier(n_jobs=-1, random_state=random_state),
            k_folds=k_folds,
            n_repetitions=n_repetitions,
            random_state=random_state,
        )
        self.m2 = CrossValidationWrapper(
            QUANTClassifier(random_state=random_state),
            k_folds=k_folds,
            n_repetitions=n_repetitions,
            random_state=random_state,
        )
        self.m3 = CrossValidationWrapper(
            RDSTClassifier(n_jobs=-1, random_state=random_state),
            k_folds=k_folds,
            n_repetitions=n_repetitions,
            random_state=random_state,
        )

        self.use_caruana = False  # !!!!!!!!!!!!!!!!!!!!!!
        # model = CatBoostClassifier(
        #    iterations=10000,
        #    early_stopping_rounds=50,
        #    learning_rate=0.0005,
        #    verbose=0
        # )

    def _fit(self, X, y):
        def add_argmax_label(df: pl.DataFrame, label_col="label"):
            numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]

            return df.with_columns(
                pl.struct(numeric_cols)
                .map_elements(lambda row: max(row, key=row.get))
                .alias(label_col)
            )

        self.train_pred1 = self.m1.fit_predict_proba(X, y)
        preds = (
            pl.DataFrame(self.train_pred1, schema=list(self.classes_))
            .pipe(add_argmax_label)["label"]
            .to_list()
        )
        acc = accuracy_score(y, preds)
        print("MR vall acc", acc)

        self.train_pred2 = self.m2.fit_predict_proba(X, y)
        preds = (
            pl.DataFrame(self.train_pred2, schema=list(self.classes_))
            .pipe(add_argmax_label)["label"]
            .to_list()
        )
        acc = accuracy_score(y, preds)
        print("QUANT vall acc", acc)

        self.train_pred3 = self.m3.fit_predict_proba(X, y)
        preds = (
            pl.DataFrame(self.train_pred3, schema=list(self.classes_))
            .pipe(add_argmax_label)["label"]
            .to_list()
        )
        acc = accuracy_score(y, preds)
        print("RDST vall acc", acc)

        # self.train_predmm1 = self.mm1.fit_predict_proba(np.hstack([self.train_pred1, self.train_pred2, self.train_pred3]), y)
        # preds = pl.DataFrame(self.train_predmm1, schema=list(self.classes_)).pipe(add_argmax_label)['label'].to_list()
        # acc = accuracy_score(y, preds)
        # print('Meta CatBoost vall acc', acc)

        if self.use_caruana:
            # reshuffle self.train_pred1 to check if Caruana works correctly
            X = np.random.rand(self.train_pred1.shape[0], self.train_pred1.shape[1])
            X = X / X.sum(axis=1, keepdims=True)
            model_predictions = {
                "MR": self.train_pred1,
                "QUANT": self.train_pred2,
                "RDST": self.train_pred3,
                "TEST": X,
            }

            def accuracy(y_true, y_pred):
                # Convert to integers (0,1,2,3) for argmax comparison
                # but *does not* affect predictions
                unique = sorted(set(y_true))
                mapping = {label: i for i, label in enumerate(unique)}
                y_idx = np.array([mapping[x] for x in y_true])

                y_hat = np.argmax(y_pred, axis=1)
                return np.mean(y_hat == y_idx)

            from amltk.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana

            self.weights, traj, final_pred = weighted_ensemble_caruana(
                model_predictions=model_predictions,
                targets=y,
                size=50,  # ensemble size / num of draws
                metric=accuracy,
                select=max,
            )
            print(self.weights)

        else:
            X_meta = np.hstack([self.train_pred1, self.train_pred2, self.train_pred3])
            self.meta_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", RidgeClassifierCV(alphas=np.logspace(-4, 4, 10))),
                ]
            )
            self.meta_model.fit(X_meta, y)

    def _predict(self, X):
        self.test_pred1 = self.m1._predict_proba(X)
        self.test_pred2 = self.m2._predict_proba(X)
        self.test_pred3 = self.m3._predict_proba(X)

        # self.test_predmm1 = self.mm1._predict_proba(np.hstack([self.test_pred1, self.test_pred2, self.test_pred3]))

        if self.use_caruana:
            X = np.random.rand(self.test_pred1.shape[0], self.test_pred1.shape[1])
            X = X / X.sum(axis=1, keepdims=True)

            model_predictions = {
                "MR": self.test_pred1,
                "QUANT": self.test_pred2,
                "RDST": self.test_pred3,
                "TEST": X,
            }

            final_probs = np.zeros((len(X), len(self.classes_)), dtype=float)
            for model_id, weight in self.weights.items():
                final_probs += weight * model_predictions[model_id]
            predicted_indices = np.argmax(final_probs, axis=1)
            return self.classes_[predicted_indices]
        else:
            X_meta = np.hstack([self.test_pred1, self.test_pred2, self.test_pred3])
            return self.meta_model.predict(X_meta)


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


class StackerV2(BaseClassifier):
    def __init__(self, random_state=None, n_repetitions="auto", k_folds="auto"):
        super().__init__()
        self.n_repetitions = n_repetitions
        self.k_folds = k_folds
        self.random_state = random_state

        self.use_caruana = False  # !!!!!!!!!!!!!!!!!!!!!!

    def _fit(self, X, y):
        n_samples = X.shape[0]

        if self.n_repetitions == "auto":
            if n_samples < 100:
                self.computes_n_repetitions = 4
            elif n_samples < 500:
                self.computes_n_repetitions = 2
            else:
                self.computes_n_repetitions = 1

        if self.k_folds == "auto":
            if n_samples < 35:
                self.computes_k_folds = n_samples
            elif n_samples < 150:
                self.computes_k_folds = 10
            else:
                self.computes_k_folds = 8

        print(f"Using {self.computes_k_folds} folds and {self.computes_n_repetitions} repetitions")

        # self.m1 = CrossValidationWrapper(
        #    MultiRocketHydraClassifier(n_jobs=-1, random_state=self.random_state),
        #    k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        # )

        # self.m4 = CrossValidationWrapper(
        #    aeon_make_pipeline(
        #        transformers.RankTransform(),
        #        MultiRocketHydraClassifier(n_jobs=-1, random_state=self.random_state+1)
        #    ),
        #    k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        # )

        # self.m2 = CrossValidationWrapper(
        #    QUANTClassifier(random_state=self.random_state),
        #    k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        # )

        # self.m5 = CrossValidationWrapper(
        #    aeon_make_pipeline(
        #        transformers.RankTransform(),
        #        QUANTClassifier(random_state=self.random_state+1)
        #    ),
        #    k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        # )

        # self.m3 = CrossValidationWrapper(
        #    RDSTClassifier(n_jobs=-1, random_state=self.random_state),
        #    k_folds=self.computes_k_folds, n_repetitions=self.computes_n_repetitions, random_state=self.random_state
        # )
        from aeon.transformations.collection.shapelet_based import (
            RandomDilatedShapeletTransform,
        )
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.pipeline import make_pipeline

        features1 = MultiRocketHydra(n_jobs=-1, random_state=self.random_state)
        models1 = make_pipeline(
            DualScaler(hydra_scaler=SparseScaler(), rocket_scaler=StandardScaler()),
            old_models.RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
        )
        features2 = QUANTTransformer()
        models2 = ExtraTreesClassifier(
            n_estimators=200,
            max_features=0.1,
            criterion="entropy",
            random_state=self.random_state,
        )

        features3 = RandomDilatedShapeletTransform(
            max_shapelets=10000,
            shapelet_lengths=None,
            proba_normalization=0.8,
            threshold_percentiles=None,
            alpha_similarity=0.5,
            use_prime_dilations=False,
            n_jobs=-1,
            random_state=self.random_state,
        )
        models3 = make_pipeline(
            StandardScaler(with_mean=True),
            old_models.RidgeClassifierCVWithProba(
                alphas=np.logspace(-4, 4, 20),
            ),
        )

        features4 = aeon_make_pipeline(
            transformers.RankTransform(),
            MultiRocketHydra(n_jobs=-1, random_state=self.random_state),
        )
        models4 = make_pipeline(
            StandardScaler(), old_models.RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10))
        )
        features5 = aeon_make_pipeline(transformers.RankTransform(), QUANTTransformer())
        models5 = ExtraTreesClassifier(
            n_estimators=200,
            max_features=0.1,
            criterion="entropy",
            random_state=self.random_state,
        )

        self.m1 = FeatureCrossValidationWrapper(
            features=features1,
            model=models1,
            k_folds=self.computes_k_folds,
            n_repetitions=self.computes_n_repetitions,
            random_state=self.random_state,
        )

        self.m2 = FeatureCrossValidationWrapper(
            features=features2,
            model=models2,
            k_folds=self.computes_k_folds,
            n_repetitions=self.computes_n_repetitions,
            random_state=self.random_state,
        )
        self.m3 = FeatureCrossValidationWrapper(
            features=features3,
            model=models3,
            k_folds=self.computes_k_folds,
            n_repetitions=self.computes_n_repetitions,
            random_state=self.random_state,
        )

        self.m4 = FeatureCrossValidationWrapper(
            features=features4,
            model=models4,
            k_folds=self.computes_k_folds,
            n_repetitions=self.computes_n_repetitions,
            random_state=self.random_state,
        )
        self.m5 = FeatureCrossValidationWrapper(
            features=features5,
            model=models5,
            k_folds=self.computes_k_folds,
            n_repetitions=self.computes_n_repetitions,
            random_state=self.random_state,
        )

        def add_argmax_label(df: pl.DataFrame, label_col="label"):
            numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]

            return df.with_columns(
                pl.struct(numeric_cols)
                .map_elements(lambda row: max(row, key=row.get))
                .alias(label_col)
            )

        self.train_pred1 = self.m1.fit_predict_proba(X, y)
        preds = (
            pl.DataFrame(self.train_pred1, schema=list(self.classes_))
            .pipe(add_argmax_label)["label"]
            .to_list()
        )
        acc = accuracy_score(y, preds)
        print("MR vall acc", acc)

        self.train_pred2 = self.m2.fit_predict_proba(X, y)
        preds = (
            pl.DataFrame(self.train_pred2, schema=list(self.classes_))
            .pipe(add_argmax_label)["label"]
            .to_list()
        )
        acc = accuracy_score(y, preds)
        print("QUANT vall acc", acc)

        self.train_pred3 = self.m3.fit_predict_proba(X, y)
        preds = (
            pl.DataFrame(self.train_pred3, schema=list(self.classes_))
            .pipe(add_argmax_label)["label"]
            .to_list()
        )
        acc = accuracy_score(y, preds)
        print("RDST vall acc", acc)

        self.train_pred4 = self.m4.fit_predict_proba(X, y)
        preds = (
            pl.DataFrame(self.train_pred4, schema=list(self.classes_))
            .pipe(add_argmax_label)["label"]
            .to_list()
        )
        acc = accuracy_score(y, preds)
        print("Ranked MR vall acc", acc)

        self.train_pred5 = self.m5.fit_predict_proba(X, y)
        preds = (
            pl.DataFrame(self.train_pred5, schema=list(self.classes_))
            .pipe(add_argmax_label)["label"]
            .to_list()
        )
        acc = accuracy_score(y, preds)
        print("Ranked QUANT vall acc", acc)

        if self.use_caruana:
            X = np.random.rand(self.train_pred1.shape[0], self.train_pred1.shape[1])
            X = X / X.sum(axis=1, keepdims=True)
            model_predictions = {
                "MR": self.train_pred1,
                "QUANT": self.train_pred2,
                "RDST": self.train_pred3,
                "TEST": X,
            }

            def accuracy(y_true, y_pred):
                unique = sorted(set(y_true))
                mapping = {label: i for i, label in enumerate(unique)}
                y_idx = np.array([mapping[x] for x in y_true])

                y_hat = np.argmax(y_pred, axis=1)
                return np.mean(y_hat == y_idx)

            from amltk.ensembling.weighted_ensemble_caruana import weighted_ensemble_caruana

            self.weights, traj, final_pred = weighted_ensemble_caruana(
                model_predictions=model_predictions,
                targets=y,
                size=50,  # ensemble size / num of draws
                metric=accuracy,
                select=max,
            )
            print(self.weights)

        else:
            X_meta = np.hstack(
                [
                    self.train_pred1,
                    self.train_pred2,
                    self.train_pred3,
                    self.train_pred4,
                    self.train_pred5,
                ]
            )
            self.meta_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", RidgeClassifierCV(alphas=np.logspace(-4, 4, 10))),
                ]
            )
            self.meta_model.fit(X_meta, y)

    def _predict(self, X):
        self.test_pred1 = self.m1._predict_proba(X)
        self.test_pred2 = self.m2._predict_proba(X)
        self.test_pred3 = self.m3._predict_proba(X)
        self.test_pred4 = self.m4._predict_proba(X)
        self.test_pred5 = self.m5._predict_proba(X)

        # self.test_predmm1 = self.mm1._predict_proba(np.hstack([self.test_pred1, self.test_pred2, self.test_pred3]))

        if self.use_caruana:
            X = np.random.rand(self.test_pred1.shape[0], self.test_pred1.shape[1])
            X = X / X.sum(axis=1, keepdims=True)

            model_predictions = {
                "MR": self.test_pred1,
                "QUANT": self.test_pred2,
                "RDST": self.test_pred3,
            }

            final_probs = np.zeros((len(X), len(self.classes_)), dtype=float)
            for model_id, weight in self.weights.items():
                final_probs += weight * model_predictions[model_id]
            predicted_indices = np.argmax(final_probs, axis=1)
            return self.classes_[predicted_indices]
        else:
            X_meta = np.hstack(
                [
                    self.test_pred1,
                    self.test_pred2,
                    self.test_pred3,
                    self.test_pred4,
                    self.test_pred5,
                ]
            )
            return self.meta_model.predict(X_meta)


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

    def _fit(self, X, y):
        with threadpool_limits(limits=1):
            return super().fit(X, y)


# class StackerV3(BaseClassifier):
#     def __init__(self, random_state=None, n_repetitions=1, k_folds=10):
#         super().__init__()
#         self.n_repetitions = n_repetitions
#         self.k_folds = k_folds
#         self.random_state = random_state
#         self.cv_splits = None
#
#     def get_model(self, name):
#         if name == "multirocket-ridgecv":
#             pipe = make_pipeline(
#                 MultiScaler(
#                     scalers={"hydra_": SparseScaler(), "multirocket_": StandardScaler()},
#                     verbose=False,
#                 ),
#                 RidgeClassifierCVIndicator(alphas=np.logspace(-3, 3, 10)),
#             )
#             return pipe
#         elif name == "rstsf":
#             return RSTSF(random_state=self.random_state, n_jobs=-1, n_estimators=100)
#         elif name == "quant-etc":
#             pipe = make_pipeline(
#                 MultiScaler(
#                     scalers={
#                         "quant_": NoScaler(),
#                     },
#                     verbose=False,
#                 ),
#                 ExtraTreesClassifier(
#                     n_estimators=200,
#                     max_features=0.1,
#                     criterion="entropy",
#                     random_state=self.random_state,  # pass this in
#                     n_jobs=-1,
#                 ),
#             )
#             return pipe
#         elif name == "rdst-ridgecv":
#             pipe = make_pipeline(
#                 MultiScaler(
#                     scalers={
#                         "rdst_": StandardScaler(),
#                     },
#                     verbose=False,
#                 ),
#                 RidgeClassifierCVIndicator(alphas=np.logspace(-4, 4, 20)),
#             )
#             return pipe
#         elif name == "rdst-robustscale-ridgecv":
#             pipe = make_pipeline(
#                 MultiScaler(
#                     scalers={
#                         "rdst_": RobustScaler(),
#                     },
#                     verbose=False,
#                 ),
#                 RidgeClassifierCVIndicator(alphas=np.logspace(-4, 4, 20)),
#             )
#             return pipe
#         elif name == "catch22-quant-et":
#             pipe = make_pipeline(
#                 MultiScaler(
#                     scalers={
#                         "catch22_": NoScaler(),
#                         "quant_": NoScaler(),
#                     },
#                     verbose=False,
#                 ),
#                 ExtraTreesClassifier(
#                     n_estimators=200,
#                     max_features=0.1,
#                     criterion="entropy",
#                     random_state=self.random_state,  # pass this in
#                     n_jobs=-1,
#                 ),
#             )
#             return pipe
#         elif name == "probability-linear-svc":
#             pipe = make_pipeline(
#                 MultiScaler(
#                     scalers={
#                         "proba_": StandardScaler(),
#                     },
#                     verbose=False,
#                 ),
#                 SVC(kernel="linear", probability=True, random_state=self.random_state),
#             )
#             return pipe
#         elif name == "probability-et":
#             pipe = make_pipeline(
#                 MultiScaler(
#                     scalers={
#                         "proba_": StandardScaler(),
#                     },
#                     verbose=False,
#                 ),
#                 ExtraTreesClassifier(
#                     n_estimators=500,
#                     # max_features=0.3,
#                     # criterion="entropy",
#                     random_state=self.random_state,  # pass this in
#                     n_jobs=-1,
#                     bootstrap=True,
#                 ),
#             )
#             return pipe
#         elif name == "probability-ridgecv":
#             pipe = make_pipeline(
#                 MultiScaler(
#                     scalers={
#                         "proba_": StandardScaler(),
#                     },
#                     verbose=False,
#                 ),
#                 RidgeClassifierCVIndicator(alphas=np.logspace(-3, 3, 10)),
#             )
#             return pipe
#         elif name == "probability-rf":
#             pipe = make_pipeline(
#                 MultiScaler(
#                     scalers={
#                         "proba_": StandardScaler(),
#                     },
#                     verbose=False,
#                 ),
#                 RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1),
#             )
#             return pipe
#         else:
#             raise ValueError(f"Unknown model name: {name}")
#
#     def fit_tranformers(self, X):
#         self.t1 = RandomDilatedShapeletTransform(n_jobs=-1, random_state=self.random_state)
#         self.t2 = QUANTTransformer()
#         self.t3 = MultiRocket(n_jobs=-1, random_state=self.random_state)
#         self.t4 = HydraTransformer(n_jobs=-1, random_state=self.random_state)
#         self.t5 = Catch22(n_jobs=-1)
#         self.t1.fit(X)
#         self.t2.fit(X)
#         self.t3.fit(X)
#         self.t4.fit(X)
#         self.t5.fit(X)
#
#     def transform_series(self, X):
#         start = perf_counter()
#         Xt1 = self.t1.transform(X)
#         t1_time = perf_counter() - start
#         print(f"RDST transform: {t1_time:.4f}s")
#
#         start = perf_counter()
#         Xt2 = self.t2.transform(X)
#         t2_time = perf_counter() - start
#         print(f"QUANT transform: {t2_time:.4f}s")
#
#         start = perf_counter()
#         Xt3 = self.t3.transform(X)
#         t3_time = perf_counter() - start
#         print(f"MultiRocket transform: {t3_time:.4f}s")
#
#         start = perf_counter()
#         Xt4 = self.t4.transform(X)
#         t4_time = perf_counter() - start
#         print(f"Hydra transform: {t4_time:.4f}s")
#
#         start = perf_counter()
#         Xt5 = self.t5.transform(X)
#         t5_time = perf_counter() - start
#         print(f"Catch22 transform: {t5_time:.4f}s")
#
#         return pl.DataFrame(
#             np.hstack([Xt1, Xt2, Xt3, Xt4, Xt5]),
#             schema=[f"rdst_{i}" for i in range(Xt1.shape[1])]
#             + [f"quant_{i}" for i in range(Xt2.shape[1])]
#             + [f"multirocket_{i}" for i in range(Xt3.shape[1])]
#             + [f"hydra_{i}" for i in range(Xt4.shape[1])]
#             + [f"catch22_{i}" for i in range(Xt5.shape[1])],
#         )
#
#     def _fit(self, X, y):
#         if self.cv_splits is None:
#             self.cv_splits = generate_folds(
#                 X,
#                 y,
#                 n_splits=self.k_folds,
#                 n_repetitions=self.n_repetitions,
#                 random_state=self.random_state,
#             )
#         self.fit_tranformers(X)
#         self.Xt_ = self.transform_series(X)
#         self.trained_models_ = []
#
#         self.tsc_algos = ["rstsf"]
#         self.feature_algos = ["multirocket-ridgecv", "quant-etc", "rdst-ridgecv"]
#         self.stacking_models = ["probability-ridgecv", "probability-et"]
#
#         for model_name in self.tsc_algos:
#             oof_proba = []
#             model_group = []
#             for train_idx, val_idx in tqdm(self.cv_splits):
#                 pipe = self.get_model(model_name)
#                 pipe.fit(X[train_idx], y[train_idx])
#                 proba = pipe.predict_proba(X[val_idx])
#
#                 prob_columns = []
#                 for idx, p in zip(val_idx, proba):
#                     d = {
#                         "index": idx,
#                     }
#                     for scls, prob in zip(pipe.classes_, p):
#                         k = f"proba_model0_{model_name}_class_{scls}"
#                         d[k] = prob.item()
#                         if k not in prob_columns:
#                             prob_columns.append(k)
#                     oof_proba.append(d)
#                 model_group.append(pipe)
#             self.trained_models_.append((model_name, model_group))
#             agg_probs = (
#                 pl.DataFrame(oof_proba).group_by("index").mean().sort("index").select(prob_columns)
#             )
#             self.Xt_ = pl.concat([self.Xt_, agg_probs], how="horizontal")
#
#             # for each model compute oof accuracy
#             oof_probas = agg_probs.to_numpy()
#             oof_pred_indices = np.argmax(oof_probas, axis=1)
#             oof_preds = self.classes_[oof_pred_indices]
#             oof_acc = accuracy_score(y, oof_preds)
#             print(f"Model {model_name}| CA: {oof_acc:.4f}")
#
#         for model_name in self.feature_algos:
#             oof_proba = []
#             model_group = []
#             for train_idx, val_idx in tqdm(self.cv_splits):
#                 pipe = self.get_model(model_name)
#                 pipe.fit(self.Xt_[train_idx], y[train_idx])
#                 proba = pipe.predict_proba(self.Xt_[val_idx])
#
#                 prob_columns = []
#                 for idx, p in zip(val_idx, proba):
#                     d = {
#                         "index": idx,
#                     }
#                     for scls, prob in zip(pipe.classes_, p):
#                         k = f"proba_model0_{model_name}_class_{scls}"
#                         d[k] = prob.item()
#                         if k not in prob_columns:
#                             prob_columns.append(k)
#                     oof_proba.append(d)
#                 model_group.append(pipe)
#             self.trained_models_.append((model_name, model_group))
#             agg_probs = (
#                 pl.DataFrame(oof_proba).group_by("index").mean().sort("index").select(prob_columns)
#             )
#             self.Xt_ = pl.concat([self.Xt_, agg_probs], how="horizontal")
#
#             # for each model compute oof accuracy
#             oof_probas = agg_probs.to_numpy()
#             oof_pred_indices = np.argmax(oof_probas, axis=1)
#             oof_preds = self.classes_[oof_pred_indices]
#             oof_acc = accuracy_score(y, oof_preds)
#             # ocmute also log loss
#             # from sklearn.metrics import log_loss, roc_auc_score
#             # log_loss_value = log_loss(y, oof_probas)
#             # ocmpute AUC
#             # if len(np.unique(y)) == 2:
#             #    auc_value = roc_auc_score(y, oof_probas[:, 1])
#             # else:
#             #    auc_value = roc_auc_score(y, oof_probas, multi_class='ovr', average='macro')
#             # print(f"Model {model_name}| CA: {oof_acc:.4f}, Log Loss: {log_loss_value:.4f}, AUC: {auc_value:.4f}")
#             print(f"Model {model_name}| CA: {oof_acc:.4f}")
#
#         for model_name in self.stacking_models:
#             oof_proba = []
#             model_group = []
#             for train_idx, val_idx in tqdm(self.cv_splits):
#                 pipe = self.get_model(model_name)
#                 pipe.fit(self.Xt_[train_idx], y[train_idx])
#                 proba = pipe.predict_proba(self.Xt_[val_idx])
#
#                 prob_columns = []
#                 for idx, p in zip(val_idx, proba):
#                     d = {
#                         "index": idx,
#                     }
#                     for scls, prob in zip(pipe.classes_, p):
#                         k = f"proba_model1_{model_name}_class_{scls}"
#                         d[k] = prob.item()
#                         if k not in prob_columns:
#                             prob_columns.append(k)
#                     oof_proba.append(d)
#                 model_group.append(pipe)
#             self.trained_models_.append((model_name, model_group))
#             agg_probs = (
#                 pl.DataFrame(oof_proba).group_by("index").mean().sort("index").select(prob_columns)
#             )
#             self.Xt_ = pl.concat([self.Xt_, agg_probs], how="horizontal")
#
#             oof_probas = agg_probs.to_numpy()
#             oof_pred_indices = np.argmax(oof_probas, axis=1)
#             oof_preds = self.classes_[oof_pred_indices]
#             oof_acc = accuracy_score(y, oof_preds)
#
#             print(f"Model {model_name}| CA: {oof_acc:.4f}")
#
#         stats_df = []
#         for model in self.tsc_algos + self.feature_algos + self.stacking_models:
#             stack = "0" if model in self.tsc_algos + self.feature_algos else "1"
#             model_columns = [
#                 col
#                 for col in self.Xt_.columns
#                 if col.startswith(f"proba_model{stack}_{model}_class_")
#             ]
#             # print(model_columns)
#             oof_proba = self.Xt_.select(model_columns)
#             # print(oof_proba)
#             oof_pred_indices = np.argmax(oof_proba, axis=1)
#             oof_preds = self.classes_[oof_pred_indices]
#             oof_acc = accuracy_score(y, oof_preds)
#             # print(f"{model}:{oof_acc:.4f}")
#             stats_df.append(
#                 {
#                     "model": model,
#                     "stack": stack,
#                     "oof_acc": oof_acc,
#                 }
#             )
#
#         stats_df = (
#             pl.DataFrame(stats_df)
#             .with_columns(
#                 pl.when(pl.col("model") == "probability-ridgecv")
#                 .then(pl.lit(1))
#                 .otherwise(pl.lit(0))
#                 .alias("is_preferred")
#             )
#             .filter(pl.col("stack") == "1")
#             .sort(["oof_acc", "stack", "is_preferred"], descending=True)
#         )
#         # print(stats_df)
#
#         self.best_model = (stats_df.row(0, named=True))["model"]
#         print(f"Best model selected for prediction: {self.best_model}")
#
#     def predict_proba_per_model(self, X):
#         Xt = self.transform_series(X)
#         return_dict = {}
#         for model_name, model_group in self.trained_models_:
#             oof_proba = []
#             for model in model_group:
#                 if model_name in self.tsc_algos:
#                     proba = model.predict_proba(X)
#                 else:
#                     proba = model.predict_proba(Xt)
#                 prob_columns = []
#                 for i, p in enumerate(proba):
#                     d = {
#                         "index": i,
#                     }
#                     for scls, prob in zip(model.classes_, p):
#                         model_stack_number = (
#                             "0" if model_name in self.tsc_algos + self.feature_algos else "1"
#                         )
#                         k = f"proba_model{model_stack_number}_{model_name}_class_{scls}"
#                         d[k] = prob.item()
#                         if k not in prob_columns:
#                             prob_columns.append(k)
#                     oof_proba.append(d)
#             df = pl.DataFrame(oof_proba).group_by("index").mean().sort("index").select(prob_columns)
#             Xt = pl.concat([Xt, df], how="horizontal")
#             return_dict[model_name] = df.to_numpy()
#         return return_dict
#
#     def _predict_proba(self, X):
#         return self.predict_proba_per_model(X)[self.best_model]
#
#     def _predict(self, X):
#         probas = self._predict_proba(X)
#         predicted_indices = np.argmax(probas, axis=1)
#         return self.classes_[predicted_indices]
