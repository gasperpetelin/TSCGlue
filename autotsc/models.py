from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter

import numpy as np
import polars as pl
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

from autotsc import transformers, utils


class AutoTSCModel(BaseClassifier):
    # TODO: change capability tags
    _tags = {
        "capability:multivariate": True,
        "capability:train_estimate": True,
        "capability:contractable": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(self, n_jobs=1, n_gpus=0, verbose=0, n_folds = 20, model_selection=None):
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

    def get_default_models(self, random_seed, model_selection="fast"):
        # add also scaled model versions
        m1 = (
            "tab-ridge",
            CrossValidationWrapper(
                SklearnClassifierWrapper(RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)))
            ),
        )
        m2 = (
            "tab-rf",
            CrossValidationWrapper(
                SklearnClassifierWrapper(RandomForestClassifier(n_jobs=-1, n_estimators=500))
            ),
        )
        m3 = ("c22", CrossValidationWrapper(Catch22Classifier(n_jobs=1)))
        m4 = (
            "minirocket",
            CrossValidationWrapper(
                MiniRocketClassifier(
                    n_jobs=1,
                    estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
                    random_state=random_seed,
                )
            ),
        )
        m5 = (
            "quant",
            CrossValidationWrapper(
                aeon_make_pipeline(transformers.CumSum(), QUANTClassifier(random_state=random_seed))
            ),
        )
        m6 = (
            "hydra",
            CrossValidationWrapper(HydraClassifier(n_jobs=1, random_state=random_seed)),
            # TODO add RidgeClassifierCVWithProba() not regular ridge
        )
        m7 = (
            "rocket",
            CrossValidationWrapper(
                RocketClassifier(
                    n_jobs=1,
                    estimator=RidgeClassifierCVWithProba(alphas=np.logspace(-3, 3, 10)),
                    random_state=random_seed,
                )
            ),
        )
        m8 = (
            "dif-quant",
            CrossValidationWrapper(
                aeon_make_pipeline(
                    transformers.Difference(), QUANTClassifier(random_state=random_seed)
                )
            ),
        )
        m9 = ("cs-quant", CrossValidationWrapper(QUANTClassifier(random_state=random_seed)))
        m10 = (
            "multirocket",
            CrossValidationWrapper(
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
                CrossValidationWrapper(
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
                CrossValidationWrapper(
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
            ("weasel", CrossValidationWrapper(WEASEL_V2(n_jobs=1, max_feature_count=3000))),
            (
                "tsf",
                CrossValidationWrapper(SupervisedTimeSeriesForest(n_jobs=1, n_estimators=20)),
            ),
            ("rstsf", CrossValidationWrapper(RSTSF(n_jobs=1, n_estimators=20))),
            ("summry", CrossValidationWrapper(SummaryClassifier(n_jobs=1))),
            (
                "rochydra",
                CrossValidationWrapper(
                    MultiRocketHydraClassifier(n_jobs=1, random_state=random_seed + 1)
                ),
            ),
            (
                "cboss",
                CrossValidationWrapper(ContractableBOSS(n_jobs=1, time_limit_in_minutes=1.0)),
            ),
            (
                "drcif",
                CrossValidationWrapper(
                    DrCIFClassifier(n_jobs=1, random_state=random_seed, time_limit_in_minutes=1.0)
                ),
            ),
            # CrossValidationWrapper(RISTClassifier(n_jobs=1)),
            # CrossValidationWrapper(RDSTClassifier(n_jobs=-1)),
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

        return [model1, model8]#, model7]

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
        default_models = self.get_default_models(
            random_seed, model_selection=self.model_selection
        )

        # Use ProcessPoolExecutor for parallel model training
        with ProcessPoolExecutor(max_workers=self.cpus_to_use_) as executor:
            futures = {}
            for model_id, model in default_models:
                future = executor.submit(
                    run_fit_predict_proba_wrapper, model_id, model, X, y, self.folds_
                )
                futures[future] = model_id

            for future in as_completed(futures):
                model_id, model, pred = future.result()
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
                        f"Trained base model {model_id}, OOF accuracy: {acc:.4f} "
                        f"in {model.fit_time_mean_:.2f}s"
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
            X_meta = self.oof_predictions_.to_numpy(writable=True)
            X_meta = np.asarray(X_meta, dtype=np.float64, order="C").copy()
            pred = model.fit_predict_proba_cv(X_meta, y, self.folds_)
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
                    f"Trained meta model {model_id}, OOF accuracy: {acc:.4f} "
                    f"in {model.fit_time_mean_:.2f}s"
                )

        return self

    def summary(self):
        return pl.DataFrame(self.summary_)

    def predict_per_model(self, X):
        "make predictions for each model in the ensemble"
        all_preds = {}
        oof_predictions_ = None

        # Use ProcessPoolExecutor for parallel predictions
        with ProcessPoolExecutor(max_workers=self.cpus_to_use_) as executor:
            futures = {}
            for model_id, model in reversed(list(self.models_.items())):
                future = executor.submit(run_predict_proba_wrapper, model_id, model, X)
                futures[future] = model_id

            for future in as_completed(futures):
                model_id, model, pred_probs = future.result()
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
            X_meta = oof_predictions_.select(self.oof_predictions_.columns).to_numpy(writable=True)
            X_meta = np.asarray(X_meta, dtype=np.float64, order="C").copy()
            pred = model.predict(X_meta)
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


class CrossValidationWrapper(BaseClassifier):
    def __init__(self, model, n_workers=None):
        super().__init__()
        self.model = model
        self.n_workers = n_workers
        self.trained_models_ = []
        self.fit_time_ = []
        self.fit_time_mean_ = None
        self.predict_time_ = []
        self.predict_time_mean_ = None

    def _fit(self, X, y):
        raise NotImplementedError()

    def _predict_proba(self, X):
        predictions = []
        # Use ProcessPoolExecutor for parallel prediction
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(run_predict_proba, model, X)
                for model in self.trained_models_
            ]
            for future in as_completed(futures):
                proba, pred_time = future.result()
                predictions.append(proba)
                self.predict_time_.append(pred_time)
        self.predict_time_mean_ = np.mean(self.predict_time_)
        avg_proba = np.mean(predictions, axis=0)
        return avg_proba

    def _predict(self, X):
        predictions = []
        # Use ProcessPoolExecutor for parallel prediction
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [
                executor.submit(run_predict_proba, model, X)
                for model in self.trained_models_
            ]
            for future in as_completed(futures):
                proba, pred_time = future.result()
                predictions.append(proba)
                self.predict_time_.append(pred_time)
        self.predict_time_mean_ = np.mean(self.predict_time_)
        avg_proba = np.mean(predictions, axis=0)
        predicted_indices = np.argmax(avg_proba, axis=1)
        return self.classes_[predicted_indices]

    def _fit_predict_proba(self, X, y, cv_splits):
        n_classes = len(np.unique(y))
        oof_proba = np.zeros((len(y), n_classes))

        # Use ProcessPoolExecutor for parallel fold training
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for train_idx, val_idx in cv_splits:
                model_clone = clone(self.model)
                future = executor.submit(run_model_on_fold, model_clone, X, y, train_idx, val_idx)
                futures.append(future)

            for future in as_completed(futures):
                model, proba, fit_time = future.result()
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
    def __init__(self, model, n_workers=None):
        self.model = model
        self.n_workers = n_workers
        self.trained_models_ = []
        self.fit_time_ = []
        self.fit_time_mean_ = None

    def fit_predict_proba_cv(self, X, y, cv_splits):
        n_classes = len(np.unique(y))
        oof_proba = np.zeros((len(y), n_classes))

        # Use ProcessPoolExecutor for parallel fold training
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for train_idx, val_idx in cv_splits:
                model_clone = clone(self.model)
                future = executor.submit(run_model_on_fold, model_clone, X, y, train_idx, val_idx)
                futures.append(future)

            for future in as_completed(futures):
                model, proba, fit_time = future.result()
                self.trained_models_.append(model)
                for idx, p in proba:
                    oof_proba[idx] = p
                self.fit_time_.append(fit_time)
        self.fit_time_mean_ = np.mean(self.fit_time_)
        return oof_proba

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


# Helper functions for parallel execution (replaces Ray remote functions)
def run_model_on_fold(model_clone, X, y, train_idx, valid_idx):
    """Train a model on a single fold and return predictions."""
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid = X[valid_idx]
    start_time = perf_counter()
    model_clone.fit(X_train, y_train)
    fit_time = perf_counter() - start_time
    proba = model_clone.predict_proba(X_valid)
    val_probs = list(zip(valid_idx, proba))
    return model_clone, val_probs, fit_time


def run_predict_proba(model, X):
    """Run predict_proba on a model."""
    start_time = perf_counter()
    proba = model.predict_proba(X)
    pred_time = perf_counter() - start_time
    return proba, pred_time


def run_fit_predict_proba_wrapper(model_id, wrapper, X, y, folds):
    """Fit a wrapper model and return predictions."""
    result = wrapper.fit_predict_proba(X, y, cv_splits=folds)
    return model_id, wrapper, result


def run_predict_proba_wrapper(model_id, wrapper, X):
    """Run predict_proba on a wrapper model."""
    result = wrapper.predict_proba(X)
    return model_id, wrapper, result
