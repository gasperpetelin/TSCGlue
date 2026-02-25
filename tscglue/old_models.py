from time import perf_counter

import numpy as np
import polars as pl
import ray
from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based import (
    HydraClassifier,
    MiniRocketClassifier,
    MultiRocketClassifier,
    MultiRocketHydraClassifier,
    RocketClassifier,
)
from aeon.classification.dictionary_based import WEASEL_V2, ContractableBOSS
from aeon.classification.feature_based import (
    Catch22Classifier,
    SummaryClassifier,
)
from aeon.classification.interval_based import (
    RSTSF,
    DrCIFClassifier,
    QUANTClassifier,
    SupervisedTimeSeriesForest,
)
from aeon.classification.sklearn import SklearnClassifierWrapper
from aeon.pipeline import make_pipeline as aeon_make_pipeline
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.transformations.collection.interval_based import QUANTTransformer
from aeon.transformations.collection.shapelet_based import (
    RandomDilatedShapeletTransform,
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

import multiprocessing
import os
import pickle
import shutil
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

from tscglue import transformers, utils
from tscglue.models import (
    DictMultiScaler,
    MultiScaler,
    NoScaler,
    RidgeClassifierCVDecisionProba,
    RidgeClassifierCVIndicator,
    SparseScaler,
    generate_folds,
    get_model_v6,
)


class AutoTSCModel(BaseClassifier):
    # TODO: change capability tags
    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(self, n_jobs=1, n_gpus=0, verbose=0, n_folds=20, model_selection=None):
        # TODO Correctlx set resource usage
        self.models_ = {}
        self.meta_models_ = {}
        self.summary_ = []
        self.oof_predictions_ = None
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.n_gpus = n_gpus
        self.model_selection = model_selection
        self.n_folds = n_folds
        self.folds_ = None
        super().__init__()

    def get_default_ray_models(self, random_seed, model_selection="fast"):
        # add also scaled model versions
        m1 = (
            "tab-ridge",
            RayCrossValidationWrapper(
                SklearnClassifierWrapper(RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)))
            ),
        )
        m2 = (
            "tab-rf",
            RayCrossValidationWrapper(
                SklearnClassifierWrapper(RandomForestClassifier(n_jobs=-1, n_estimators=500))
            ),
        )
        m3 = ("c22", RayCrossValidationWrapper(Catch22Classifier(n_jobs=1)))
        m4 = (
            "minirocket",
            RayCrossValidationWrapper(
                MiniRocketClassifier(
                    n_jobs=1,
                    estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
                    random_state=random_seed,
                )
            ),
        )
        m5 = (
            "quant",
            RayCrossValidationWrapper(
                aeon_make_pipeline(transformers.CumSum(), QUANTClassifier(random_state=random_seed))
            ),
        )
        m6 = (
            "hydra",
            RayCrossValidationWrapper(HydraClassifier(n_jobs=1, random_state=random_seed)),
            # TODO add RidgeClassifierCVWithProba() not regular ridge
        )
        m7 = (
            "rocket",
            RayCrossValidationWrapper(
                RocketClassifier(
                    n_jobs=1,
                    estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
                    random_state=random_seed,
                )
            ),
        )
        m8 = (
            "dif-quant",
            RayCrossValidationWrapper(
                aeon_make_pipeline(
                    transformers.Difference(), QUANTClassifier(random_state=random_seed)
                )
            ),
        )
        m9 = ("cs-quant", RayCrossValidationWrapper(QUANTClassifier(random_state=random_seed)))
        m10 = (
            "multirocket",
            RayCrossValidationWrapper(
                MultiRocketClassifier(
                    n_jobs=1,
                    estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
                    random_state=random_seed,
                )
            ),
        )
        if model_selection == "fast":
            return [m1, m2]  # , m3, m4, m5, m6, m7, m8, m9, m10]
        return [
            m1,
            m2,
            (
                "dif-roc",
                RayCrossValidationWrapper(
                    aeon_make_pipeline(
                        transformers.Difference(),
                        RocketClassifier(
                            n_jobs=1,
                            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
                            random_state=random_seed,
                        ),
                    )
                ),
            ),
            (
                "cs-roc",
                RayCrossValidationWrapper(
                    aeon_make_pipeline(
                        transformers.CumSum(),
                        MultiRocketClassifier(
                            n_jobs=1,
                            estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
                            random_state=random_seed,
                        ),
                    )
                ),
            ),
            m3,
            m7,
            m4,
            m5,
            m8,
            m9,
            m6,
            m10,
            ("weasel", RayCrossValidationWrapper(WEASEL_V2(n_jobs=1, max_feature_count=3000))),
            (
                "tsf",
                RayCrossValidationWrapper(SupervisedTimeSeriesForest(n_jobs=1, n_estimators=20)),
            ),
            ("rstsf", RayCrossValidationWrapper(RSTSF(n_jobs=1, n_estimators=20))),
            ("summry", RayCrossValidationWrapper(SummaryClassifier(n_jobs=1))),
            (
                "rochydra",
                RayCrossValidationWrapper(
                    MultiRocketHydraClassifier(n_jobs=1, random_state=random_seed + 1)
                ),
            ),
            (
                "cboss",
                RayCrossValidationWrapper(ContractableBOSS(n_jobs=1, time_limit_in_minutes=1.0)),
            ),
            (
                "drcif",
                RayCrossValidationWrapper(
                    DrCIFClassifier(n_jobs=1, random_state=random_seed, time_limit_in_minutes=1.0)
                ),
            ),
            # RayCrossValidationWrapper(RISTClassifier(n_jobs=1)),
            # RayCrossValidationWrapper(RDSTClassifier(n_jobs=-1)),
            # REDCOMETS(n_jobs=-1, n_trees=50)
        ]

    def get_default_metamodels(self):
        model1 = ("m-ridgecv", Ensemble(RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10))))
        # model2 = ("m-rf", Ensemble(RandomForestClassifier(n_jobs=-1, n_estimators=500)))
        # model3 = ("m-hgb", Ensemble(HistGradientBoostingClassifier()))
        # model4 = ("m-et", Ensemble(ExtraTreesClassifier(n_jobs=-1, n_estimators=500)))
        # model5 = (
        #    "m-rf-pruned",
        #    Ensemble(RandomForestClassifier(n_jobs=-1, n_estimators=500, ccp_alpha=0.01)),
        # )
        # model6 = (
        #    "m-pca-rf",
        #    Ensemble(
        #        make_pipeline(
        #            StandardScaler(),
        #            PCA(n_components=0.95),
        #            RandomForestClassifier(n_jobs=-1, n_estimators=500, ccp_alpha=0.01),
        #        )
        #    ),
        # )

        model7 = ("m-svm", Ensemble(SVC(kernel="linear", probability=True)))
        model8 = ("m-log", Ensemble(LogisticRegression(n_jobs=-1)))

        return [model1]  # , model8]#, model7]

    def _fit(self, X, y):
        self.cpus_available_, self.cpus_to_use_, self.gpus_available_, self.gpus_to_use_ = (
            utils.get_resource_config(self.n_jobs, self.n_gpus)
        )
        random_seed = np.random.randint(0, 10000)
        if self.verbose > 0:
            utils.print_fit_start_info(
                X,
                y,
                self.cpus_to_use_,
                self.cpus_available_,
                self.gpus_to_use_,
                self.gpus_available_,
                random_seed,
                n_folds=self.n_folds,
            )
        self.folds_ = utils.get_folds(X, y, n_splits=self.n_folds)
        # default_models = self.get_default_models()
        default_ray_models = self.get_default_ray_models(
            random_seed, model_selection=self.model_selection
        )

        ts = []
        # create X and y reference
        X_ref = ray.put(X)
        y_ref = ray.put(y)
        for model_id, model in default_ray_models:
            t = ray_run_fit_predict_proba_wrapper.remote(model_id, model, X_ref, y_ref, self.folds_)
            ts.append(t)

        ray_results = ray.get(ts)
        for model_id, model, pred in ray_results:
            self.models_[model_id] = model
            pred_max = np.argmax(pred, axis=1)
            acc = accuracy_score(y, model.classes_[pred_max])
            self.summary_.append(
                {
                    "model_id": model_id,
                    "model": repr(model).replace("\n", "").replace(" ", ""),
                    "validation_accuracy": acc,
                    "stacking_level": 0,
                    "train_time": model.fit_time_mean_,
                }
            )

            if self.verbose > 0:
                print(
                    f"Trained base model {model_id}, OOF accuracy: {acc:.4f} in {model.fit_time_mean_:.2f}s"
                )

            columns = [f"model_{model_id}__class_{l}" for l in list(model.classes_)]
            if self.oof_predictions_ is None:
                self.oof_predictions_ = pl.DataFrame(pred, schema=columns)
            else:
                df_pred = pl.DataFrame(pred, schema=columns)
                self.oof_predictions_ = pl.concat(
                    [self.oof_predictions_, df_pred], how="horizontal"
                )

        default_metamodels = self.get_default_metamodels()
        for model_id, model in default_metamodels:
            X = self.oof_predictions_.to_numpy(writable=True)
            X = np.asarray(X, dtype=np.float64, order="C").copy()
            pred = model.fit_predict_proba_cv(X, y, self.folds_)
            pred_max = np.argmax(pred, axis=1)
            acc = accuracy_score(y, model.trained_models_[0].classes_[pred_max])
            self.meta_models_[model_id] = model
            self.summary_.append(
                {
                    "model_id": model_id,
                    "model": repr(model).replace("\n", "").replace(" ", ""),
                    "validation_accuracy": acc,
                    "stacking_level": 1,
                    "train_time": model.fit_time_mean_,
                }
            )
            if self.verbose > 0:
                print(
                    f"Trained meta model {model_id}, OOF accuracy: {acc:.4f} in {model.fit_time_mean_:.2f}s"
                )

        return self

    def summary(self):
        return pl.DataFrame(self.summary_)

    def predict_per_model(self, X):
        "make predictions for each model in the ensemble"
        all_preds = {}
        oof_predictions_ = None
        tasks = []
        X_ref = ray.put(X)
        for model_id, model in reversed(list(self.models_.items())):
            t = ray_run_predict_proba_wrapper.remote(model_id, model, X_ref)
            tasks.append(t)

        # for model_id, model in self.models_.items():
        for model_id, model, pred_probs in ray.get(tasks):
            # pred_probs = model.predict_proba(X)
            columns = [f"model_{model_id}__class_{l}" for l in list(model.classes_)]
            if oof_predictions_ is None:
                oof_predictions_ = pl.DataFrame(pred_probs, schema=columns)
            else:
                df_pred = pl.DataFrame(pred_probs, schema=columns)
                oof_predictions_ = pl.concat([oof_predictions_, df_pred], how="horizontal")
            pred = np.argmax(pred_probs, axis=1)
            pred_labels = model.classes_[pred]
            all_preds[model_id] = pred_labels
            if self.verbose > 0:
                print(f"Prediction with {model_id} made in {model.predict_time_mean_:.2f}s")

        for model_id, model in self.meta_models_.items():
            X = oof_predictions_.select(self.oof_predictions_.columns).to_numpy(writable=True)
            X = np.asarray(X, dtype=np.float64, order="C").copy()
            pred = model.predict(X)
            all_preds[model_id] = pred

        return all_preds

    def _predict(self, X):
        best_model_id = self.summary().sort("validation_accuracy").tail(1)["model_id"].item()
        if self.verbose > 0:
            print(f"Using best model {best_model_id} for predictions")
        predictions = self.predict_per_model(X)
        return predictions[best_model_id]


class RidgeClassifierCVWithProba(RidgeClassifierCV):
    def predict_proba(self, X):
        return self._predict_proba_lr(X)


class RayCrossValidationWrapper(BaseClassifier):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.trained_models_ = []
        self.fit_time_ = []
        self.fit_time_mean_ = None
        self.predict_time_ = []
        self.predict_time_mean_ = None

    def _fit(self, X, y):
        raise NotImplementedError()

    def _predict_proba(self, X):
        predictions = []
        tasks = [ray_run_predict_proba.remote(model, X) for model in self.trained_models_]
        for task in tasks:
            proba, pred_time = ray.get(task)
            predictions.append(proba)
            self.predict_time_.append(pred_time)
        self.predict_time_mean_ = np.mean(self.predict_time_)
        avg_proba = np.mean(predictions, axis=0)
        return avg_proba

    def _predict(self, X):
        tasks = [ray_run_predict_proba.remote(model, X) for model in self.trained_models_]
        predictions = []
        for task in tasks:
            proba, pred_time = ray.get(task)
            predictions.append(proba)
            self.predict_time_.append(pred_time)
        self.predict_time_mean_ = np.mean(self.predict_time_)
        avg_proba = np.mean(predictions, axis=0)
        predicted_indices = np.argmax(avg_proba, axis=1)
        return self.classes_[predicted_indices]

    def _fit_predict_proba(self, X, y, cv_splits):
        n_classes = len(np.unique(y))
        oof_proba = np.zeros((len(y), n_classes))

        fold_tasks = []
        for train_idx, val_idx in cv_splits:
            model_clone = clone(self.model)
            task = ray_run_model_on_fold.remote(model_clone, X, y, train_idx, val_idx)
            fold_tasks.append(task)

        results = ray.get(fold_tasks)
        for model, proba, fit_time in results:
            self.trained_models_.append(model)
            for idx, p in proba:
                oof_proba[idx] = p
            self.fit_time_.append(fit_time)
        self.fit_time_mean_ = np.mean(self.fit_time_)
        return oof_proba

    def fit_predict_proba(self, X, y, cv_splits):
        X, y, single_class = self._fit_setup(X, y)
        y_proba = self._fit_predict_proba(X, y, cv_splits)
        self.is_fitted = True
        return y_proba


class Ensemble:
    def __init__(self, model):
        self.model = model
        self.trained_models_ = []
        self.fit_time_ = []
        self.fit_time_mean_ = None

    def fit_predict_proba_cv(self, X, y, cv_splits):
        n_classes = len(np.unique(y))
        oof_proba = np.zeros((len(y), n_classes))

        fold_tasks = []
        for train_idx, val_idx in cv_splits:
            pass
            model_clone = clone(self.model)
            task = ray_run_model_on_fold.remote(model_clone, X, y, train_idx, val_idx)
            fold_tasks.append(task)

        results = ray.get(fold_tasks)
        for model, proba, fit_time in results:
            self.trained_models_.append(model)
            for idx, p in proba:
                oof_proba[idx] = p
            self.fit_time_.append(fit_time)
        self.fit_time_mean_ = np.mean(self.fit_time_)
        return oof_proba

        # proba_predictions = []
        # for train_idx, val_idx in cv_splits:
        #    model_clone = clone(self.model)
        #    model_clone.fit(X[train_idx], y[train_idx])
        #    self.models.append(model_clone)
        #    y_proba = model_clone.predict_proba(X[val_idx])
        #    proba_predictions.extend(zip(val_idx, y_proba))
        # proba_predictions = sorted(proba_predictions)
        # return np.array([proba for idx, proba in proba_predictions])

    def predict(self, X):
        # Create a truly writable copy for SVM's C code
        X = np.array(X, dtype=np.float64, order="C", copy=True)
        X.setflags(write=True)
        predictions = np.array([model.predict_proba(X) for model in self.trained_models_])
        avg_proba = predictions.mean(axis=0)
        predicted_indices = np.argmax(avg_proba, axis=1)
        return self.trained_models_[0].classes_[predicted_indices]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.model})"


class EnsambleWeights:
    def __init__(self):
        pass


@ray.remote(num_cpus=1)
def ray_run_model_on_fold(model_clone, X, y, train_idx, valid_idx):
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    start_time = perf_counter()
    model_clone.fit(X_train, y_train)
    fit_time = perf_counter() - start_time
    proba = model_clone.predict_proba(X_valid)
    val_probs = list(zip(valid_idx, proba))
    # print(f"Train model {model_clone.__class__.__name__} on fold done in {fit_time:.2f}s")
    return model_clone, val_probs, fit_time


@ray.remote(num_cpus=1)
def ray_run_predict_proba(model, X):
    start_time = perf_counter()
    proba = model.predict_proba(X)
    pred_time = perf_counter() - start_time
    # print(f"Predict proba with model {model.__class__.__name__} done in {pred_time:.2f}s")
    return proba, pred_time


@ray.remote(num_cpus=0, resources={"meta": 1})
def ray_run_fit_predict_proba_wrapper(model_id, wrapper, X, y, folds):
    start_fp = perf_counter()
    result = wrapper.fit_predict_proba(X, y, cv_splits=folds)
    total_fp_time = perf_counter() - start_fp
    # print(f"Completed fit_predict_proba in Ray task in {total_fp_time:.2f}s")
    return model_id, wrapper, result


@ray.remote(num_cpus=0, resources={"meta": 1})
def ray_run_predict_proba_wrapper(model_id, wrapper, X):
    result = wrapper.predict_proba(X)
    return model_id, wrapper, result


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
                        "probability|model0": NoScaler(),
                    },
                    verbose=False,
                ),
                ExtraTreesClassifier(
                    n_estimators=500,
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
    elif name == "multirockethydra-p-ridgecv":
        pipe = make_pipeline(
            MultiScaler(
                scalers={
                    "feature|hydra": SparseScaler(),
                    "feature|multirocket": StandardScaler(),
                },
                verbose=False,
            ),
            RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10)),
        )
        return pipe
    # elif name == "all-ridgecv":
    #     pipe = make_pipeline(
    #         MultiScaler(
    #             scalers={
    #                 "hydra_": SparseScaler(),
    #                 "multirocket_": StandardScaler(),
    #                 "rdst_": StandardScaler(),
    #             },
    #             verbose=False,
    #         ),
    #         RidgeClassifierCVIndicator(alphas=np.logspace(-3, 3, 10)),
    #     )
    #     return pipe
    elif name == "rstsf":
        return RSTSF(random_state=seed, n_jobs=n_jobs, n_estimators=100)
    # elif name == "drcif":
    #     return DrCIFClassifier(random_state=seed, n_jobs=n_jobs, time_limit_in_minutes=2)
    # elif name == "weasel-v2":
    #     return WEASEL_V2(random_state=seed, n_jobs=n_jobs)
    # elif name == "contractable-boss":
    #     return ContractableBOSS(random_state=seed, n_jobs=n_jobs, time_limit_in_minutes=0.1)
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
    elif name == "rdst-p-ridgecv":
        pipe = make_pipeline(
            MultiScaler(
                scalers={
                    "feature|randomdilatedshapelet": StandardScaler(),
                },
                verbose=False,
            ),
            RidgeClassifierCVDecisionProba(alphas=np.logspace(-4, 4, 20)),
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
    # elif name == "rdst-robustscale-ridgecv":
    #     pipe = make_pipeline(
    #         MultiScaler(
    #             scalers={
    #                 "rdst_": RobustScaler(),
    #             },
    #             verbose=False,
    #         ),
    #         RidgeClassifierCVIndicator(alphas=np.logspace(-4, 4, 20)),
    #     )
    #     return pipe
    # elif name == "catch22-quant-et":
    #     pipe = make_pipeline(
    #         MultiScaler(
    #             scalers={
    #                 "catch22_": NoScaler(),
    #                 "quant_": NoScaler(),
    #             },
    #             verbose=False,
    #         ),
    #         ExtraTreesClassifier(
    #             n_estimators=200,
    #             max_features=0.1,
    #             criterion="entropy",
    #             random_state=seed,
    #             n_jobs=n_jobs,
    #         ),
    #     )
    #     return pipe
    # elif name == "probability-linear-svc":
    #     pipe = make_pipeline(
    #         MultiScaler(
    #             scalers={
    #                 "proba_": StandardScaler(),
    #             },
    #             verbose=False,
    #         ),
    #         SVC(kernel="linear", probability=True, random_state=seed),
    #     )
    #     return pipe
    elif name == "probability-et":
        pipe = make_pipeline(
            MultiScaler(
                scalers={
                    "probability|model0": NoScaler(),
                },
                verbose=False,
            ),
            ExtraTreesClassifier(
                n_estimators=1000,
                random_state=seed,
                n_jobs=n_jobs,
                #bootstrap=True,
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
                    "probability|model0": NoScaler(),
                },
                verbose=False,
            ),
            RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        )
        return pipe
    else:
        raise ValueError(f"Unknown model name: {name}")


# Global worker data - loaded once per worker process via initializer
# Each worker subprocess has its own copy, freed when executor shuts down
_loky_worker_data = None


def _init_worker_train(X_path, Xt_path, Xt_columns_path, y_path):
    """Initialize worker by loading data from mmap files. Called once per worker."""
    global _loky_worker_data
    import os
    pid = os.getpid()
    #print(f"[Worker {pid}] Train init starting...", flush=True)
    #print(f"[Worker {pid}] Loading X...", flush=True)
    X = np.load(X_path, mmap_mode='r')
    #print(f"[Worker {pid}] Loading Xt...", flush=True)
    Xt_np = np.load(Xt_path, mmap_mode='r')
    #print(f"[Worker {pid}] Loading Xt_columns...", flush=True)
    with open(Xt_columns_path, 'rb') as f:
        Xt_columns = pickle.load(f)
    #print(f"[Worker {pid}] Creating DataFrame ({len(Xt_columns)} cols)...", flush=True)
    Xt = pl.DataFrame(Xt_np, schema=Xt_columns)
    #print(f"[Worker {pid}] Loading y...", flush=True)
    y = np.load(y_path, mmap_mode='r')
    _loky_worker_data = (X, Xt, y)
    #print(f"[Worker {pid}] Train init done. X={X.shape}, Xt={Xt.shape}, y={y.shape}", flush=True)


def _init_worker_predict(X_path, Xt_path, Xt_columns_path):
    """Initialize worker for prediction by loading data from mmap files."""
    global _loky_worker_data
    import os
    pid = os.getpid()
    #print(f"[Worker {pid}] Predict init starting...", flush=True)
    #print(f"[Worker {pid}] Loading X...", flush=True)
    X = np.load(X_path, mmap_mode='r')
    #print(f"[Worker {pid}] Loading Xt...", flush=True)
    Xt_np = np.load(Xt_path, mmap_mode='r')
    #print(f"[Worker {pid}] Loading Xt_columns...", flush=True)
    with open(Xt_columns_path, 'rb') as f:
        Xt_columns = pickle.load(f)
    #print(f"[Worker {pid}] Creating DataFrame ({len(Xt_columns)} cols)...", flush=True)
    Xt = pl.DataFrame(Xt_np, schema=Xt_columns)
    _loky_worker_data = (X, Xt, None)
    #print(f"[Worker {pid}] Predict init done. X={X.shape}, Xt={Xt.shape}", flush=True)


def _train_one_model(fold_number, model_name, use_Xt, train_idx, val_idx, model_seed):
    """One-shot training function - uses pre-loaded data from initializer."""
    global _loky_worker_data
    import os
    pid = os.getpid()
    #print(f"[Worker {pid}] Task start: {model_name} fold {fold_number}", flush=True)
    X, Xt, y = _loky_worker_data

    #print(f"[Worker {pid}] Getting model {model_name}...", flush=True)
    pipe = get_model(model_name, seed=model_seed)
    start_train = perf_counter()

    #print(f"[Worker {pid}] Fitting {model_name}...", flush=True)
    if use_Xt:
        pipe.fit(Xt[train_idx], y[train_idx])
        #print(f"[Worker {pid}] Predicting {model_name}...", flush=True)
        proba = pipe.predict_proba(Xt[val_idx])
    else:
        pipe.fit(X[train_idx], y[train_idx])
        #print(f"[Worker {pid}] Predicting {model_name}...", flush=True)
        proba = pipe.predict_proba(X[val_idx])

    train_dur = perf_counter() - start_train
    #print(f"[Worker {pid}] Task done: {model_name} fold {fold_number} in {train_dur:.2f}s", flush=True)
    return (train_idx, val_idx, proba, pickle.dumps(pipe), train_dur, model_name, fold_number)


def _predict_one_model(model_name, pickle_pipe, use_series):
    """One-shot prediction function - uses pre-loaded data from initializer."""
    global _loky_worker_data
    import os
    pid = os.getpid()
    #print(f"[Worker {pid}] Predict task start: {model_name}", flush=True)
    X, Xt, _ = _loky_worker_data

    #print(f"[Worker {pid}] Unpickling model...", flush=True)
    pipe = pickle.loads(pickle_pipe)
    start_predict = perf_counter()

    #print(f"[Worker {pid}] Running predict_proba...", flush=True)
    if use_series:
        proba = pipe.predict_proba(X)
    else:
        proba = pipe.predict_proba(Xt)

    predict_dur = perf_counter() - start_predict
    #print(f"[Worker {pid}] Predict task done: {model_name} in {predict_dur:.2f}s", flush=True)
    return (proba, pipe.classes_, predict_dur, model_name)


def _load_feature_dict(feature_manifest):
    """Load feature arrays from mmap files and concatenate per type."""
    feature_dict = {}
    for feat_type, paths in feature_manifest.items():
        arrays = [np.load(p, mmap_mode='r') for p in paths]
        feature_dict[feat_type] = np.hstack(arrays) if len(arrays) > 1 else arrays[0]
    return feature_dict


def _train_one_model_v6(fold_number, model_name, is_series, train_idx, val_idx, model_seed,
                         X_path, y_path, feature_manifest, save_path):
    """Training function for V6 - loads mmap data per task, saves model to disk."""
    X = np.load(X_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')
    feature_dict = _load_feature_dict(feature_manifest)

    scaler, clf = get_model_v6(model_name, seed=model_seed)
    start_train = perf_counter()

    if is_series:
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[val_idx])
        with open(save_path, 'wb') as f:
            pickle.dump((None, clf), f)
    else:
        train_dict = {k: v[train_idx] for k, v in feature_dict.items()}
        val_dict = {k: v[val_idx] for k, v in feature_dict.items()}
        X_train = scaler.fit_transform(train_dict)
        X_val = scaler.transform(val_dict)
        clf.fit(X_train, y[train_idx])
        proba = clf.predict_proba(X_val)
        with open(save_path, 'wb') as f:
            pickle.dump((scaler, clf), f)

    model_size = os.path.getsize(save_path)
    train_dur = perf_counter() - start_train
    return (train_idx, val_idx, proba, clf.classes_, model_size, train_dur, model_name, fold_number)


def _predict_one_model_v6(model_name, model_path, is_series, X_path, feature_manifest):
    """Prediction function for V6 - loads model from disk, loads mmap data per task."""
    X = np.load(X_path, mmap_mode='r')
    feature_dict = _load_feature_dict(feature_manifest)

    with open(model_path, 'rb') as f:
        scaler, clf = pickle.load(f)
    start_predict = perf_counter()

    if is_series:
        proba = clf.predict_proba(X)
    else:
        X_scaled = scaler.transform(feature_dict)
        proba = clf.predict_proba(X_scaled)

    predict_dur = perf_counter() - start_predict
    return (proba, clf.classes_, predict_dur, model_name)


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

        self.feature_models = ["multirockethydra-ridgecv", "quant-etc", "rdst-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []
        self.stacking_models = ["probability-ridgecv"]
        self.best_model = "probability-ridgecv"


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


class LokyStackerV5SoftET(LokyStackerV5):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, time_limit_in_seconds=None, n_jobs=1):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, time_limit_in_seconds=time_limit_in_seconds, n_jobs=n_jobs)

        self.feature_models = ["multirockethydra-p-ridgecv", "quant-etc", "rdst-p-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []


        stacking_model = "probability-et"
        self.stacking_models = [stacking_model]
        self.best_model = stacking_model


class LokyStackerV5SoftRidge(LokyStackerV5):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, time_limit_in_seconds=None, n_jobs=1):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, time_limit_in_seconds=time_limit_in_seconds, n_jobs=n_jobs)

        self.feature_models = ["multirockethydra-p-ridgecv", "quant-etc", "rdst-p-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []


        stacking_model = "probability-ridgecv"
        self.stacking_models = [stacking_model]
        self.best_model = stacking_model


class LokyStackerV5SoftRF(LokyStackerV5):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, time_limit_in_seconds=None, n_jobs=1):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, time_limit_in_seconds=time_limit_in_seconds, n_jobs=n_jobs)

        self.feature_models = ["multirockethydra-p-ridgecv", "quant-etc", "rdst-p-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []


        stacking_model = "probability-rf"
        self.stacking_models = [stacking_model]
        self.best_model = stacking_model


class LokyStackerV6(LokyStackerV5):
    """Like LokyStackerV5 but stores features as per-type numpy arrays instead of a single polars DataFrame.

    Workers receive a dict of mmap arrays keyed by feature type, and DictMultiScaler
    selects which arrays each model needs based on its scalers dict.
    """
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, time_limit_in_seconds=None, n_jobs=1, keep_features=False):
        super().__init__(
            random_state=random_state, n_repetitions=n_repetitions,
            k_folds=k_folds, time_limit_in_seconds=time_limit_in_seconds, n_jobs=n_jobs,
        )
        self._feature_manifest = {}  # dict: str -> list[str], feature type -> list of .npy paths
        self._run_id = None
        self._base_dir = None  # ./tscglue/<run_id>
        self._model_dir = None  # ./tscglue/<run_id>/models
        self._tmpdir = None  # ./tscglue/<run_id>/features
        self.keep_features = keep_features  # If True, don't delete feature arrays after fit

    def calculate_features(self, feature_type: str, X: np.ndarray, repetition: int):
        transform = self.get_next_feature_transformer(feature_type=feature_type, n_jobs=self.n_jobs)
        transform.fit(X)
        X_t = transform.transform(X)
        self.feature_transformers.append((feature_type, transform))

        # Dump directly to disk
        path = f"{self._tmpdir}/Xt_{feature_type}_r{repetition}.npy"
        arr = np.asarray(X_t, dtype=np.float64)
        np.save(path, arr)

        if feature_type not in self._feature_manifest:
            self._feature_manifest[feature_type] = []
        self._feature_manifest[feature_type].append(path)
        return arr.shape, os.path.getsize(path) / (1024 * 1024)

    def _build_probability_array(self, n_samples):
        """Build a numpy array from level-0 OOF predictions and save as mmap.

        Returns the probability array (n_samples, n_prob_columns).
        """
        level0_preds = [p for p in self.predictions if p["level"] == 0]
        if len(level0_preds) == 0:
            return None
        df = (
            pl.DataFrame(level0_preds)
            .pivot(
                values="probability",
                index="index",
                on=["level", "model", "class"],
                aggregate_function="mean",
            )
            .sort("index")
        )
        # Sort probability columns for deterministic ordering across training/prediction
        prob_cols = sorted(c for c in df.columns if c != "index")
        self._probability_columns = prob_cols
        prob_array = df.select(prob_cols).to_numpy()
        return prob_array

    def _compute_oof_accuracy(self, y, model_name):
        """Compute OOF accuracy for a given model from self.predictions."""
        model_preds = [p for p in self.predictions if p["model"] == model_name]
        if len(model_preds) == 0:
            return 0.0
        df = (
            pl.DataFrame(model_preds)
            .pivot(
                values="probability",
                index="index",
                on=["level", "model", "class"],
                aggregate_function="mean",
            )
            .sort("index")
        )
        probas = df.drop("index").to_numpy()
        pred_indices = np.argmax(probas, axis=1)
        preds = self.classes_[pred_indices]
        return accuracy_score(y, preds)

    def _fit(self, X, y):
        import shutil
        fit_start_time = perf_counter()

        self.predictions = []
        self.trained_models_ = []
        self._feature_manifest = {}
        self.feature_transformers = []

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

        # Create run directory: ./tscglue/<run_id>/{models,features}
        if self._base_dir and os.path.exists(self._base_dir):
            shutil.rmtree(self._base_dir)
        self._run_id = uuid.uuid4().hex[:16]
        self._base_dir = os.path.join(".", "tscglue_runs", self._run_id)
        self._model_dir = os.path.join(self._base_dir, "models")
        self._tmpdir = os.path.join(self._base_dir, "features")
        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._tmpdir, exist_ok=True)
        print(f"Starting executor with {self.n_jobs} workers, run_dir={self._base_dir}")

        quant_start_time = perf_counter()
        shape, size_mb = self.calculate_features(feature_type="quant", X=X, repetition=0)
        quant_durration = perf_counter() - quant_start_time
        print(
            f"[{perf_counter() - fit_start_time:.4f}s] Computed QUANT features {shape} ({size_mb:.2f} MB) in {quant_durration:.4f}s"
        )

        X_path = f"{self._tmpdir}/X.npy"
        y_path = f"{self._tmpdir}/y.npy"
        np.save(X_path, X)
        np.save(y_path, y)

        # Accumulate all splits across repetitions for stacking
        all_splits = []

        try:
            for repetition in range(self.n_repetitions):
                print(f"[{perf_counter() - fit_start_time:.4f}s] Starting repetition {repetition}")

                multirocket_start_time = perf_counter()
                shape, size_mb = self.calculate_features(feature_type="multirocket", X=X, repetition=repetition)
                multirocket_durration = perf_counter() - multirocket_start_time
                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] Computed MultiRocket features {shape} ({size_mb:.2f} MB) in {multirocket_durration:.4f}s"
                )

                hydra_start_time = perf_counter()
                shape, size_mb = self.calculate_features(feature_type="hydra", X=X, repetition=repetition)
                hydra_durration = perf_counter() - hydra_start_time
                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] Computed Hydra features {shape} ({size_mb:.2f} MB) in {hydra_durration:.4f}s"
                )

                rdst_start_time = perf_counter()
                shape, size_mb = self.calculate_features(feature_type="rdst", X=X, repetition=repetition)
                rdst_durration = perf_counter() - rdst_start_time
                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] Computed RDST features {shape} ({size_mb:.2f} MB) in {rdst_durration:.4f}s"
                )

                current_splits = generate_folds(
                    X, y, n_splits=self.k_folds, n_repetitions=1, random_state=self._get_feature_seed()
                )
                all_splits.extend(current_splits)

                feature_manifest = dict(self._feature_manifest)

                # Build list of tasks for this repetition
                tasks = []
                for model_name in self.feature_models + self.series_models:
                    is_series = model_name in self.series_models
                    for fold_number, (train_idx, val_idx) in enumerate(current_splits):
                        model_seed = self._get_feature_seed()
                        save_path = f"{self._model_dir}/{model_name}_r{repetition}_f{fold_number}.pkl"
                        tasks.append((fold_number, model_name, is_series, train_idx, val_idx, model_seed,
                                      X_path, y_path, feature_manifest, save_path))

                n_workers = min(self.n_jobs, len(tasks))
                print(f"[{perf_counter() - fit_start_time:.4f}s] Starting training with {n_workers} workers for {len(tasks)} models")

                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=multiprocessing.get_context('spawn'),
                ) as executor:
                    futures = {
                        executor.submit(_train_one_model_v6, *task): task
                        for task in tasks
                    }

                    model_groups = {}
                    for future in as_completed(futures):
                        task = futures[future]
                        fold_number, model_name = task[0], task[1]
                        save_path = task[-1]
                        try:
                            result = future.result()
                        except Exception as e:
                            raise RuntimeError(f"Worker failed during training {model_name} fold {fold_number}: {e}")

                        train_idx, val_idx, proba, classes_, model_size, train_dur, model_name, fold_number = result
                        print(
                            f"[{perf_counter() - fit_start_time:.4f}s] Trained {model_name} in {train_dur:.4f}s for f-{fold_number}/r-{repetition} ({model_size / (1024 * 1024):.2f} MB)"
                        )

                        level = 0 if model_name in self.feature_models + self.series_models else 1
                        for idx, p in zip(val_idx, proba):
                            for scls, prob in zip(classes_, p):
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
                        model_groups[model_name].append(save_path)

                        if len(model_groups[model_name]) == self.k_folds:
                            print(
                                f"[{perf_counter() - fit_start_time:.4f}s] Completed training for model {model_name}"
                            )

                            self.trained_models_.append((model_name, model_groups[model_name]))
                            del model_groups[model_name]

                            oof_acc = self._compute_oof_accuracy(y, model_name)
                            print(
                                f"[{perf_counter() - fit_start_time:.4f}s] OOF acc for model {model_name}: {oof_acc}"
                            )

                print(f"[{perf_counter() - fit_start_time:.4f}s] Completed repetition {repetition}")

            # Train stacking models only once after all repetitions
            print(f"[{perf_counter() - fit_start_time:.4f}s] Starting stacking model training (single pass)")

            # Build probability array from level-0 predictions
            prob_array = self._build_probability_array(n_samples=X.shape[0])

            # Check for NaN values
            if prob_array is None or np.isnan(prob_array).any():
                print(f"[{perf_counter() - fit_start_time:.4f}s] NaN values detected in probability array, skipping stacking")
                print(f"[{perf_counter() - fit_start_time:.4f}s] Falling back to MultiRocketHydraClassifier")
                self.fallback_model = MultiRocketHydraClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
                self.fallback_model.fit(X, y)
                print(f"[{perf_counter() - fit_start_time:.4f}s] Fallback model trained successfully")
                return

            # Save probability array and update manifest
            prob_path = f"{self._tmpdir}/Xt_probabilities.npy"
            np.save(prob_path, prob_array)
            stacking_manifest = {"probabilities": [prob_path]}

            X_path, y_path = f"{self._tmpdir}/X.npy", f"{self._tmpdir}/y.npy"

            for model_name in self.stacking_models:
                tasks = []
                is_series = model_name in self.series_models
                for fold_number, (train_idx, val_idx) in enumerate(all_splits):
                    model_seed = self._get_feature_seed()
                    save_path = f"{self._model_dir}/{model_name}_stacking_f{fold_number}.pkl"
                    tasks.append((fold_number, model_name, is_series, train_idx, val_idx, model_seed,
                                  X_path, y_path, stacking_manifest, save_path))

                n_workers = min(self.n_jobs, len(tasks))
                with ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=multiprocessing.get_context('spawn'),
                ) as executor:
                    futures = {
                        executor.submit(_train_one_model_v6, *task): task
                        for task in tasks
                    }

                    model_group = []
                    for future in as_completed(futures):
                        task = futures[future]
                        fold_number, model_name_task = task[0], task[1]
                        save_path = task[-1]
                        try:
                            result = future.result()
                        except Exception as e:
                            raise RuntimeError(f"Worker failed during stacking training {model_name_task} fold {fold_number}: {e}")

                        train_idx, val_idx, proba, classes_, model_size, train_dur, model_name_result, fold_number = result

                        print(
                            f"[{perf_counter() - fit_start_time:.4f}s] Trained {model_name_result} in {train_dur:.4f}s for f-{fold_number} ({model_size / (1024 * 1024):.2f} MB)"
                        )

                        level = 1
                        for idx, p in zip(val_idx, proba):
                            for scls, prob in zip(classes_, p):
                                d = {
                                    "index": idx,
                                    "model": model_name_result,
                                    "repetition": 0,
                                    "level": level,
                                    "class": scls.item(),
                                    "probability": prob.item(),
                                }
                                self.predictions.append(d)
                        model_group.append(save_path)

                self.trained_models_.append((model_name, model_group))

                oof_acc = self._compute_oof_accuracy(y, model_name)
                print(
                    f"[{perf_counter() - fit_start_time:.4f}s] OOF acc for model {model_name}: {oof_acc}"
                )

            print(f"[{perf_counter() - fit_start_time:.4f}s] Completed all repetitions and stacking")

        finally:
            # Clean up temp directory unless keep_features is set
            if not self.keep_features and self._tmpdir and os.path.exists(self._tmpdir):
                shutil.rmtree(self._tmpdir)
                self._tmpdir = None
            print("Executor shutdown complete")

    def compute_features(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Compute features for prediction, returning a dict of numpy arrays keyed by feature type."""
        feature_dict = {}
        for feature_type, transform in self.feature_transformers:
            X_t = transform.transform(X)
            if feature_type not in feature_dict:
                feature_dict[feature_type] = []
            feature_dict[feature_type].append(X_t)
        return {k: np.hstack(v) if len(v) > 1 else v[0] for k, v in feature_dict.items()}

    def predict_proba_per_model(self, X):
        import shutil
        predict_start_time = perf_counter()
        print(f"[{perf_counter() - predict_start_time:.4f}s] Starting prediction")

        # Create features directory for mmap files
        self._tmpdir = os.path.join(self._base_dir, "features")
        os.makedirs(self._tmpdir, exist_ok=True)
        print(f"Starting executor with {self.n_jobs} workers, run_dir={self._base_dir}")

        try:
            feature_dict = self.compute_features(X)
            print(f"[{perf_counter() - predict_start_time:.4f}s] Computed features for prediction")

            # Save feature arrays to mmap
            X_path = f"{self._tmpdir}/X.npy"
            np.save(X_path, X)
            feature_manifest = {}
            for feat_type, arr in feature_dict.items():
                path = f"{self._tmpdir}/Xt_{feat_type}.npy"
                np.save(path, arr)
                feature_manifest[feat_type] = [path]
            print(f"[{perf_counter() - predict_start_time:.4f}s] Feature arrays saved to mmap files")

            predictions = []

            first_level_models = [(model_name, model_group) for model_name, model_group in self.trained_models_
                                  if model_name in self.feature_models + self.series_models]

            # Build tasks for first-level models
            tasks = []
            for model_name, model_paths in reversed(first_level_models):
                is_series = model_name in self.series_models
                for model_path in model_paths:
                    tasks.append((model_name, model_path, is_series,
                                  X_path, feature_manifest))

            n_workers = min(self.n_jobs, len(tasks))
            print(f"[{perf_counter() - predict_start_time:.4f}s] Starting prediction with {n_workers} workers for {len(tasks)} first-level models")

            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context('spawn'),
            ) as executor:
                futures = {
                    executor.submit(_predict_one_model_v6, *task): task
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

            # Build probability array from level-0 predictions for stacking
            if self._tmpdir and os.path.exists(self._tmpdir):
                shutil.rmtree(self._tmpdir)
            self._tmpdir = os.path.join(self._base_dir, "features")
            os.makedirs(self._tmpdir, exist_ok=True)

            # Pivot level-0 predictions into probability array
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
            # Use same sorted column order as training to ensure scaler alignment
            prob_cols = sorted(c for c in df.columns if c != "index")
            prob_array = df.select(prob_cols).to_numpy()

            X_path = f"{self._tmpdir}/X.npy"
            np.save(X_path, X)
            prob_path = f"{self._tmpdir}/Xt_probabilities.npy"
            np.save(prob_path, prob_array)
            stacking_manifest = {"probabilities": [prob_path]}

            stacking_models = [(model_name, model_group) for model_name, model_group in self.trained_models_
                               if model_name in self.stacking_models]

            # Build tasks for stacking models
            tasks = []
            for model_name, model_paths in reversed(stacking_models):
                is_series = model_name in self.series_models
                for model_path in model_paths:
                    tasks.append((model_name, model_path, is_series,
                                  X_path, stacking_manifest))

            n_workers = min(self.n_jobs, len(tasks))
            print(f"[{perf_counter() - predict_start_time:.4f}s] Starting prediction with {n_workers} workers for {len(tasks)} stacking models")

            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context('spawn'),
            ) as executor:
                futures = {
                    executor.submit(_predict_one_model_v6, *task): task
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

            # Build return dict: average probabilities per model across folds
            all_preds_df = (
                pl.DataFrame(predictions)
                .pivot(
                    values="probability",
                    index="index",
                    on=["level", "model", "class"],
                    aggregate_function="mean",
                )
                .sort("index")
            )
            return_dict = {}
            for model_name, _ in self.trained_models_:
                prob_columns = sorted(col for col in all_preds_df.columns if model_name in col)
                agg_probs = all_preds_df.select(prob_columns)
                return_dict[model_name] = agg_probs.to_numpy()

            return return_dict

        finally:
            # Clean up temp directory
            if self._tmpdir and os.path.exists(self._tmpdir):
                shutil.rmtree(self._tmpdir)
                self._tmpdir = None
            print("Executor shutdown complete")

    def cleanup(self):
        """Remove saved models and features from disk."""
        import shutil
        if self._base_dir and os.path.exists(self._base_dir):
            shutil.rmtree(self._base_dir)
        self._base_dir = None
        self._model_dir = None
        self._tmpdir = None
        self._run_id = None


class LokyStackerV6SoftET(LokyStackerV6):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, time_limit_in_seconds=None, n_jobs=1):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, time_limit_in_seconds=time_limit_in_seconds, n_jobs=n_jobs)

        self.feature_models = ["multirockethydra-p-ridgecv", "quant-etc", "rdst-p-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []

        stacking_model = "probability-et"
        self.stacking_models = [stacking_model]
        self.best_model = stacking_model


class LokyStackerV6SoftRidge(LokyStackerV6):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, time_limit_in_seconds=None, n_jobs=1):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, time_limit_in_seconds=time_limit_in_seconds, n_jobs=n_jobs)

        self.feature_models = ["multirockethydra-p-ridgecv", "quant-etc", "rdst-p-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []

        stacking_model = "probability-ridgecv"
        self.stacking_models = [stacking_model]
        self.best_model = stacking_model


class LokyStackerV6SoftRF(LokyStackerV6):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, time_limit_in_seconds=None, n_jobs=1):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, time_limit_in_seconds=time_limit_in_seconds, n_jobs=n_jobs)

        self.feature_models = ["multirockethydra-p-ridgecv", "quant-etc", "rdst-p-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []

        stacking_model = "probability-rf"
        self.stacking_models = [stacking_model]
        self.best_model = stacking_model
