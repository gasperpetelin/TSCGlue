import os
import uuid

from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif

import numpy as np
import polars as pl
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.interval_based import RSTSF
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.transformations.collection.interval_based import QUANTTransformer, RandomIntervals
from aeon.transformations.collection import ARCoefficientTransformer, PeriodogramTransformer
from aeon.utils.numba.general import first_order_differences_3d
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from aeon.transformations.collection.shapelet_based import (
    RandomDilatedShapeletTransform,
)
import pickle
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
import shutil
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from tscglue import utils
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.pipeline import Pipeline


class AutoSelectKBestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier=None, k=None, k_min=6000, k_max=35000, midpoint=300, steepness=0.010):
        self.classifier = classifier
        self.k = k
        self.k_min = k_min
        self.k_max = k_max
        self.midpoint = midpoint
        self.steepness = steepness

    def _optimal_k(self, n_train: int) -> int:
        return int(
            self.k_min
            + (self.k_max - self.k_min)
            / (1.0 + np.exp(-self.steepness * (n_train - self.midpoint)))
        )

    def fit(self, X, y):
        return self._fit(X, y)

    def predict(self, X):
        return self._predict(X)

    def predict_proba(self, X):
        return self._predict_proba(X)

    # internal helpers
    def _fit(self, X, y):
        k = self.k if self.k is not None else self._optimal_k(X.shape[0])

        if self.classifier is None:
            clf = RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10))
        else:
            clf = clone(self.classifier)

        self.classifier_ = Pipeline(
            [
                ("var", VarianceThreshold()),
                ("select", SelectKBest(score_func=f_classif, k=k)),
                ("clf", clf),
            ]
        )
        self.classifier_.fit(X, y)

        # sklearn convention: expose classes_
        inner = self.classifier_.named_steps["clf"]
        if hasattr(inner, "classes_"):
            self.classes_ = inner.classes_

        return self

    def _predict(self, X):
        return self.classifier_.predict(X)

    def _predict_proba(self, X):
        inner = self.classifier_.named_steps["clf"]
        if not hasattr(inner, "predict_proba"):
            raise AttributeError("Underlying classifier does not support predict_proba().")
        return self.classifier_.predict_proba(X)



def get_model_v6(name, seed=None, n_jobs=1):
    """Returns (DictMultiScaler, classifier) for feature/stacking models, or (None, pipe) for series models."""
    if name == "multirockethydra-ridgecv":
        scaler = DictMultiScaler(scalers={"hydra": SparseScaler(), "multirocket": StandardScaler()})
        clf = RidgeClassifierCVIndicator(alphas=np.logspace(-3, 3, 10))
        return scaler, clf
    elif name == "multirockethydra-p-ridgecv":
        scaler = DictMultiScaler(scalers={"hydra": SparseScaler(), "multirocket": StandardScaler()})
        clf = RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10))
        return scaler, clf
    elif name == "quant-etc":
        scaler = DictMultiScaler(scalers={"quant": NoScaler()})
        clf = ExtraTreesClassifier(
            n_estimators=200, max_features=0.1, criterion="entropy",
            random_state=seed, n_jobs=n_jobs,
        )
        return scaler, clf
    elif name == "rdst-ridgecv":
        scaler = DictMultiScaler(scalers={"rdst": StandardScaler()})
        clf = RidgeClassifierCVIndicator(alphas=np.logspace(-4, 4, 20))
        return scaler, clf
    elif name == "rdst-p-ridgecv":
        scaler = DictMultiScaler(scalers={"rdst": StandardScaler()})
        clf = RidgeClassifierCVDecisionProba(alphas=np.logspace(-4, 4, 20))
        return scaler, clf
    elif name == "probability-ridgecv":
        scaler = DictMultiScaler(scalers={"probabilities": StandardScaler()})
        clf = RidgeClassifierCVIndicator(alphas=np.logspace(-3, 3, 20))
        return scaler, clf
    elif name == "probability-et":
        scaler = DictMultiScaler(scalers={"probabilities": NoScaler()})
        clf = ExtraTreesClassifier(
            n_estimators=1000, random_state=seed, n_jobs=n_jobs,
        )
        return scaler, clf
    elif name == "probability-rf":
        scaler = DictMultiScaler(scalers={"probabilities": NoScaler()})
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        return scaler, clf
    elif name == "multirockethydra-bestk-p-ridgecv":
        scaler = DictMultiScaler(scalers={"hydra": SparseScaler(), "multirocket": StandardScaler()})
        clf = AutoSelectKBestClassifier()
        return scaler, clf
    elif name == "rstsf":
        return None, RSTSF(random_state=seed, n_jobs=n_jobs, n_estimators=100)
    else:
        raise ValueError(f"Unknown model name: {name}")



def _load_feature_dict_v7(directory, feature_specs):
    """Load feature arrays using read_array with (feat_type, repetition) specs."""
    feature_dict = {}
    for feat_type, rep in feature_specs.items():
        feature_dict[feat_type] = read_array(f"Xt_{feat_type}", directory, repetition=rep)
    return feature_dict


def _train_one_model_v7(fold_number, model_name, is_series, train_idx, val_idx, model_seed,
                         directory, feature_specs, model_dir, repetition):
    """Training function for V7 - loads data via read_array, saves model to disk."""
    X = read_array("X", directory)
    y = read_array("y", directory)
    feature_dict = _load_feature_dict_v7(directory, feature_specs)

    scaler, clf = get_model_v6(model_name, seed=model_seed)
    start_train = perf_counter()

    if is_series:
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[val_idx])
        _, model_size = save_model((None, clf), model_name, model_dir, repetition, fold_number)
    else:
        clf.fit(scaler.fit_transform(feature_dict, idx=train_idx), y[train_idx])
        proba = clf.predict_proba(scaler.transform(feature_dict, idx=val_idx))
        _, model_size = save_model((scaler, clf), model_name, model_dir, repetition, fold_number)

    train_dur = perf_counter() - start_train
    return (train_idx, val_idx, proba, clf.classes_, model_size, train_dur, model_name, fold_number)


def _predict_one_model_v7(display_name, model_name, is_series, directory, feature_specs, model_dir, repetition, fold):
    """Prediction function for V7 - loads model from disk, loads data via read_array."""
    X = read_array("X", directory)
    feature_dict = _load_feature_dict_v7(directory, feature_specs)

    scaler, clf = read_model(model_name, model_dir, repetition, fold)
    start_predict = perf_counter()

    if is_series:
        proba = clf.predict_proba(X)
    else:
        X_scaled = scaler.transform(feature_dict)
        proba = clf.predict_proba(X_scaled)

    predict_dur = perf_counter() - start_predict
    return (proba, clf.classes_, predict_dur, display_name)


def save_array(X, name, directory, dtype=None, repetition=None):
    if dtype is not None:
        X = np.asarray(X, dtype=dtype)
    suffix = f"_r_{repetition}" if repetition is not None else ""
    path = f"{directory}/{name}{suffix}.npy"
    np.save(path, X)
    size = os.path.getsize(path) / (1024 * 1024)
    return path, X.shape, size

def read_array(name, directory, repetition=None, mmap_mode="r", allow_pickle=False):
    suffix = f"_r_{repetition}" if repetition is not None else ""
    path = f"{directory}/{name}{suffix}.npy"
    # TODO remove allow_pickle=True once all features/models are numpy arrays
    return np.load(path, mmap_mode=mmap_mode, allow_pickle=allow_pickle)

def save_model(model, name, directory, repetition=None, fold=None):
    rep_suffix = f"_r_{repetition}" if repetition is not None else ""
    fold_suffix = f"_f_{fold}" if fold is not None else ""
    path = f"{directory}/{name}{rep_suffix}{fold_suffix}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    size = os.path.getsize(path)
    return path, size

def read_model(name, directory, repetition=None, fold=None):
    rep_suffix = f"_r_{repetition}" if repetition is not None else ""
    fold_suffix = f"_f_{fold}" if fold is not None else ""
    path = f"{directory}/{name}{rep_suffix}{fold_suffix}.pkl"
    with open(path, 'rb') as f:
        return pickle.load(f)

def get_feature_transformer(feature_type: str, seed: int, n_jobs: int = 1):
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

def _noop():
    return None

class LokyStackerV7(BaseClassifier):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10,
                 n_jobs=1, keep_features=False,
                 hyperparameters=None, verbose=0,
                 feature_models=None, series_models=None,
                 stacking_models=None):
        super().__init__()
        self.n_repetitions = n_repetitions
        self.k_folds = k_folds
        self.random_state = random_state
        self.cv_splits = None
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.feature_seed = np.random.default_rng(random_state)

        self._run_id = uuid.uuid4().hex[:16]
        self._base_dir = os.path.join(".", "tscglue_runs", self._run_id)
        self._model_dir = os.path.join(self._base_dir, "models")
        self._tmpdir = os.path.join(self._base_dir, "features_training")
        self.keep_features = keep_features

        self.feature_models = feature_models if feature_models is not None else ["multirockethydra-ridgecv", "quant-etc", "rdst-ridgecv"]
        self.series_models = series_models if series_models is not None else ["rstsf"]
        self.stacking_models = stacking_models if stacking_models is not None else ["probability-ridgecv"]
        assert len(self.stacking_models) == 1, f"Expected exactly 1 stacking model, got {len(self.stacking_models)}"
        self.best_model = self.stacking_models[-1]

        self.hyperparameters = hyperparameters
        self._oof_scores = []

    def _get_feature_seed(self):
        return int(self.feature_seed.integers(0, 2**31 - 1, dtype=np.int32))

    def log(self, message, level, start_time=None, current_time=None):
        if self.verbose >= level:
            if start_time is not None:
                if current_time is None:
                    current_time = perf_counter()
                print(f"[{current_time - start_time:.2f}s] {message}")
            else:
                print(message)

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

    def _has_fallback(self):
        fallback_path = f"{self._model_dir}/fallback.pkl"
        return self._model_dir and os.path.exists(fallback_path)

    def _predict_proba(self, X):
        if self._has_fallback():
            fallback = read_model("fallback", self._model_dir, )
            return fallback.predict_proba(X)
        return self.predict_proba_per_model(X)[self.best_model]

    def _predict(self, X):
        if self._has_fallback():
            fallback = read_model("fallback", self._model_dir, )
            return fallback.predict(X)
        probas = self._predict_proba(X)
        predicted_indices = np.argmax(probas, axis=1)
        return self.classes_[predicted_indices]

    def _on_stacking_complete(self, fit_start_time=None):
        """Hook called after all stacking models are trained. Handles auto best-model selection."""
        if self.best_model == "auto-best-stacking":
            scores = [s for s in self._oof_scores if s["level"] == 1]
        elif self.best_model == "auto-best-base":
            scores = [s for s in self._oof_scores if s["level"] == 0]
        elif self.best_model == "auto-best":
            scores = list(self._oof_scores)
        else:
            return
        if scores:
            selected_model = max(scores, key=lambda s: s["oof_accuracy"])["model"]
            # Stacking OOF entries may be repetition-tagged (e.g. "probability-ridgecv_r1"),
            # while inference keys remain unsuffixed stacking model names.
            parts = selected_model.rsplit("_r", 1)
            if len(parts) == 2 and parts[1].isdigit() and parts[0] in self.stacking_models:
                selected_model = parts[0]
            self.best_model = selected_model
            self.log(f"Auto-selected best model: {self.best_model}", level=1, start_time=fit_start_time)

    def cleanup(self):
        """Remove saved models and features from disk."""
        if self._base_dir and os.path.exists(self._base_dir):
            shutil.rmtree(self._base_dir)

    def calculate_features(self, feature_type: str, X: np.ndarray, repetition: int):
        transform = get_feature_transformer(feature_type, seed=self._get_feature_seed(), n_jobs=self.n_jobs)
        X_t = transform.fit_transform(X)
        model_path, model_size = save_model(transform, f"transformer_{feature_type}", self._model_dir, repetition)
        array_path, array_shape, array_size = save_array(X_t, f"Xt_{feature_type}", self._tmpdir, dtype=np.float64, repetition=repetition)
        return model_path, model_size, array_path, array_shape, array_size

    def _save_model_predictions(self, predictions, model_name, n_samples, level):
        """Save a model's predictions to disk and remove from the list.

        Saves:
        - {model_name}.npy: (n_samples, n_classes) array with OOF probabilities
        - {model_name}_meta.npy: [level, class_0, class_1, ...] metadata

        Returns the predictions list with this model's entries removed.
        """
        model_preds = [p for p in predictions if p["model"] == model_name]
        if not model_preds:
            return predictions

        # Get unique classes from predictions
        classes = sorted(set(p["class"] for p in model_preds))
        n_classes = len(classes)
        class_to_idx = {c: i for i, c in enumerate(classes)}

        # Build (n_samples, n_classes) array with NaN for non-OOF indices
        prob_array = np.full((n_samples, n_classes), np.nan, dtype=np.float64)
        for p in model_preds:
            prob_array[p["index"], class_to_idx[p["class"]]] = p["probability"]

        # Save probabilities and metadata to features directory
        save_array(prob_array, f"pred_{model_name}", self._tmpdir)
        save_array(np.array([level] + classes), f"pred_{model_name}_meta", self._tmpdir)

        return [p for p in predictions if p["model"] != model_name]

    def _load_model_predictions(self, model_name):
        """Load a model's predictions from disk. Returns (prob_array, level, classes)."""
        prob_array = read_array(f"pred_{model_name}", self._tmpdir)
        meta = read_array(f"pred_{model_name}_meta", self._tmpdir, allow_pickle=True, mmap_mode=None)
        level = int(meta[0])
        classes = list(meta[1:])
        return prob_array, level, classes

    def _build_probability_array(self, n_samples):
        """Build probability array from disk — no averaging since model names are unique per repetition."""
        # Load all level-0 predictions from disk (files starting with pred_ but not _meta)
        prob_files = [f for f in os.listdir(self._tmpdir) if f.startswith("pred_") and f.endswith(".npy") and not f.endswith("_meta.npy")]

        all_cols = []
        col_names = []
        for prob_file in sorted(prob_files):
            model_name = prob_file[5:-4]  # Remove "pred_" prefix and ".npy" suffix
            prob_array, level, classes = self._load_model_predictions(model_name)
            if level != 0:
                continue
            for i, cls in enumerate(classes):
                col_name = f"{level}_{model_name}_{cls}"
                col_names.append(col_name)
                all_cols.append(prob_array[:, i])

        if not all_cols:
            return None

        # Sort columns for consistent ordering
        sorted_indices = sorted(range(len(col_names)), key=lambda i: col_names[i])
        self._probability_columns = [col_names[i] for i in sorted_indices]
        prob_array = np.column_stack([all_cols[i] for i in sorted_indices])
        return prob_array

    def _compute_oof_accuracy(self, y, model_name):
        """Compute OOF accuracy for a given model from disk."""
        prob_array, level, classes = self._load_model_predictions(model_name)

        # Get non-NaN rows (OOF samples)
        valid_mask = ~np.isnan(prob_array).any(axis=1)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            return 0.0

        probas = prob_array[valid_mask]
        pred_indices = np.argmax(probas, axis=1)
        preds = np.array(classes)[pred_indices]
        return accuracy_score(y[valid_indices], preds)

    def _fit_fallback(self, X, y, fit_start_time):
        self.log("Falling back to MultiRocketHydraClassifier", level=1, start_time=fit_start_time)
        fallback = MultiRocketHydraClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        fallback.fit(X, y)
        save_model(fallback, "fallback", self._model_dir)
        self.log("Fallback model trained successfully", level=1, start_time=fit_start_time)

    def _fit(self, X, y):
        fit_start_time = perf_counter()
        self.log(f"Starting executor with {self.n_jobs} workers, run_dir={self._base_dir}", level=1, start_time=fit_start_time)
        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._tmpdir, exist_ok=True)

        start_save_x_y_time = perf_counter()
        X_path, _, _ = save_array(X, "X", self._tmpdir, dtype=np.float64) # TODO Ensure this is saved as 32 or 64 depending on the input
        y_path, _, _ = save_array(y, "y", self._tmpdir)
        save_durration = perf_counter() - start_save_x_y_time
        #print(f"[{perf_counter() - fit_start_time:.4f}s] Saved X and y to disk in {save_durration:.4f}s")

        self.log(f"Saved X and y to disk in {save_durration:.2f}s", level=2, start_time=fit_start_time)

        # Check if each class has at least 2 instances for fold training
        _, counts = np.unique(y, return_counts=True)
        if np.any(counts < 2):
            self.log("Some classes have fewer than 2 instances, fold training not possible", level=1, start_time=fit_start_time)
            self._fit_fallback(X, y, fit_start_time)
            return

        if self.cv_splits is None:
            self.cv_splits = []

        predictions = []

        quant_start_time = perf_counter()
        model_path, model_size, array_path, shape, size_mb = self.calculate_features(feature_type="quant", X=X, repetition=0)
        quant_durration = perf_counter() - quant_start_time
        self.log(f"Computed QUANT features {shape} ({size_mb:.2f} MB) in {quant_durration:.4f}s", level=1, start_time=fit_start_time)


        # Accumulate all splits across repetitions for stacking
        all_splits = []

        self._executor = ProcessPoolExecutor(
            max_workers=self.n_jobs,
            mp_context=multiprocessing.get_context('spawn'),
        )
        futures = [self._executor.submit(_noop) for _ in range(self.n_jobs)]
        with self._executor as executor:

            try:
                for repetition in range(self.n_repetitions):

                    self.log(f"Starting repetition {repetition}", level=2, start_time=fit_start_time)

                    multirocket_start_time = perf_counter()
                    _, _, _, shape, size_mb = self.calculate_features(feature_type="multirocket", X=X, repetition=repetition)
                    multirocket_durration = perf_counter() - multirocket_start_time
                    self.log(f"Computed MultiRocket features {shape} ({size_mb:.2f} MB) in {multirocket_durration:.4f}s", level=1, start_time=fit_start_time)

                    hydra_start_time = perf_counter()
                    _, _, _, shape, size_mb = self.calculate_features(feature_type="hydra", X=X, repetition=repetition)
                    hydra_durration = perf_counter() - hydra_start_time
                    self.log(f"Computed Hydra features {shape} ({size_mb:.2f} MB) in {hydra_durration:.4f}s", level=1, start_time=fit_start_time)

                    rdst_start_time = perf_counter()
                    _, _, _, shape, size_mb = self.calculate_features(feature_type="rdst", X=X.astype(np.float64), repetition=repetition)
                    rdst_durration = perf_counter() - rdst_start_time
                    self.log(f"Computed RDST features {shape} ({size_mb:.2f} MB) in {rdst_durration:.4f}s", level=1, start_time=fit_start_time)

                    current_splits = generate_folds(
                        X, y, n_splits=self.k_folds, n_repetitions=1, random_state=self._get_feature_seed()
                    )
                    all_splits.extend(current_splits)

                    # Quant is computed once at r0 and reused across all repetitions.
                    feature_specs = {ft: repetition for ft in ("quant", "multirocket", "hydra", "rdst")}
                    feature_specs["quant"] = 0

                    # Build list of tasks — model names tagged with repetition
                    tasks = []
                    for model_name in self.series_models + self.feature_models:
                        is_series = model_name in self.series_models
                        for fold_number, (train_idx, val_idx) in enumerate(current_splits):
                            model_seed = self._get_feature_seed()
                            tasks.append((fold_number, model_name, is_series, train_idx, val_idx, model_seed,
                                        self._tmpdir, feature_specs, self._model_dir, repetition))

                    n_workers = min(self.n_jobs, len(tasks))
                    self.log(f"Starting training with {n_workers} workers for {len(tasks)} models", level=2, start_time=fit_start_time)
                    
                    futures = {
                        executor.submit(_train_one_model_v7, *task): task
                        for task in tasks
                    }

                    model_groups = {}
                    for future in as_completed(futures):
                        task = futures[future]
                        fold_number, base_model_name = task[0], task[1]
                        try:
                            result = future.result()
                        except Exception as e:
                            tagged_name = f"{base_model_name}_r{repetition}"
                            raise RuntimeError(f"Worker failed during training {tagged_name} fold {fold_number}: {e}")

                        train_idx, val_idx, proba, classes_, model_size, train_dur, model_name_result, fold_number = result
                        model_name_result = f"{model_name_result}_r{repetition}"
                        self.log(f"Trained {model_name_result} in {train_dur:.4f}s for f-{fold_number}/r-{repetition} ({model_size / (1024 * 1024):.2f} MB)", level=2, start_time=fit_start_time)

                        level = 0
                        for idx, p in zip(val_idx, proba):
                            for scls, prob in zip(classes_, p):
                                d = {
                                    "index": idx,
                                    "model": model_name_result,
                                    "repetition": repetition,
                                    "level": level,
                                    "class": scls.item(),
                                    "probability": prob.item(),
                                }
                                predictions.append(d)

                        if model_name_result not in model_groups:
                            model_groups[model_name_result] = []
                        model_groups[model_name_result].append(fold_number)

                        if len(model_groups[model_name_result]) == self.k_folds:
                            self.log(f"Completed training for model {model_name_result}", level=2, start_time=fit_start_time)
                            del model_groups[model_name_result]

                            # Save OOF predictions to disk and clear from memory
                            predictions = self._save_model_predictions(predictions, model_name_result, n_samples=X.shape[0], level=0)

                            oof_acc = self._compute_oof_accuracy(y, model_name_result)
                            self._oof_scores.append({"model": model_name_result, "level": 0, "oof_accuracy": oof_acc})
                            self.log(f"OOF acc for model {model_name_result}: {oof_acc}", level=1, start_time=fit_start_time)

                    self.log(f"Completed repetition {repetition}", level=1, start_time=fit_start_time)

                # Train stacking models only once after all repetitions
                self.log("Starting stacking model training (single pass)", level=2, start_time=fit_start_time)

                # Build probability array from level-0 predictions (no averaging — model names are unique per rep)
                prob_array = self._build_probability_array(n_samples=X.shape[0])

                # Check for NaN values
                if prob_array is None or np.isnan(prob_array).any():
                    self.log("NaN values detected in probability array, skipping stacking", level=2, start_time=fit_start_time)
                    self.log("Falling back to MultiRocketHydraClassifier", level=2, start_time=fit_start_time)
                    fallback = MultiRocketHydraClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
                    fallback.fit(X, y)
                    save_model(fallback, "fallback", self._model_dir)
                    self.log("Fallback model trained successfully", level=2, start_time=fit_start_time)
                    return

                # Save probability array
                save_array(prob_array, "Xt_probabilities", self._tmpdir)
                stacking_specs = {"probabilities": None}

                for model_name in self.stacking_models:
                    tasks = []
                    is_series = model_name in self.series_models
                    for fold_number, (train_idx, val_idx) in enumerate(all_splits):
                        model_seed = self._get_feature_seed()
                        tasks.append((fold_number, model_name, is_series, train_idx, val_idx, model_seed,
                                    self._tmpdir, stacking_specs, self._model_dir, 0))

                    n_workers = min(self.n_jobs, len(tasks))
                    futures = {
                        executor.submit(_train_one_model_v7, *task): task
                        for task in tasks
                    }

                    model_groups = {}
                    for future in as_completed(futures):
                        task = futures[future]
                        fold_number, model_name_task = task[0], task[1]
                        try:
                            result = future.result()
                        except Exception as e:
                            raise RuntimeError(f"Worker failed during stacking training {model_name_task} fold {fold_number}: {e}")

                        train_idx, val_idx, proba, classes_, model_size, train_dur, model_name_result, fold_number = result
                        repetition = fold_number // self.k_folds
                        fold_in_repetition = fold_number % self.k_folds
                        tagged_model_name = f"{model_name_result}_r{repetition}"

                        self.log(
                            f"Trained {tagged_model_name} in {train_dur:.4f}s for f-{fold_in_repetition}/r-{repetition} ({model_size / (1024 * 1024):.2f} MB)",
                            level=2,
                            start_time=fit_start_time,
                        )

                        level = 1
                        for idx, p in zip(val_idx, proba):
                            for scls, prob in zip(classes_, p):
                                d = {
                                    "index": idx,
                                    "model": tagged_model_name,
                                    "repetition": repetition,
                                    "level": level,
                                    "class": scls.item(),
                                    "probability": prob.item(),
                                }
                                predictions.append(d)

                        if tagged_model_name not in model_groups:
                            model_groups[tagged_model_name] = []
                        model_groups[tagged_model_name].append(fold_number)

                        if len(model_groups[tagged_model_name]) == self.k_folds:
                            self.log(f"Completed training for model {tagged_model_name}", level=2, start_time=fit_start_time)
                            del model_groups[tagged_model_name]

                            # Save OOF predictions to disk and clear from memory
                            predictions = self._save_model_predictions(
                                predictions, tagged_model_name, n_samples=X.shape[0], level=1
                            )

                            oof_acc = self._compute_oof_accuracy(y, tagged_model_name)
                            self._oof_scores.append({"model": tagged_model_name, "level": 1, "oof_accuracy": oof_acc})
                            self.log(f"OOF acc for model {tagged_model_name}: {oof_acc}", level=1, start_time=fit_start_time)

                self.log("Completed all repetitions and stacking", level=1, start_time=fit_start_time)
                self._on_stacking_complete(fit_start_time=fit_start_time)

            finally:
                # Clean up temp directory unless keep_features is set
                if not self.keep_features and self._tmpdir and os.path.exists(self._tmpdir):
                    cleanup_start = perf_counter()
                    shutil.rmtree(self._tmpdir)
                    self.log(f"Cleaned up tmpdir in {perf_counter() - cleanup_start:.2f}s", level=2, start_time=fit_start_time)
                    self._tmpdir = None
                # Store training dir in a fitted attribute that survives aeon's post-fit reset
                if self.keep_features and self._tmpdir:
                    self.features_training_dir_ = self._tmpdir
                self.log("Executor shutdown complete", level=2, start_time=fit_start_time)

    def _get_training_dir(self):
        """Return the training features directory, checking both _tmpdir and the fitted attribute."""
        d = getattr(self, "features_training_dir_", None) or self._tmpdir
        if not self.keep_features or not d or not os.path.exists(d):
            raise RuntimeError(
                f"Not available. Set keep_features=True before fitting. "
                f"keep_features={self.keep_features}, dir={d}"
            )
        return d

    def get_oof_predictions(self) -> pl.DataFrame:
        """Return OOF predictions as a polars DataFrame.

        Requires keep_features=True.
        Returns a DataFrame with columns: model_name|class for each model/class,
        with NaN for non-OOF indices.
        """
        d = self._get_training_dir()
        frames = []
        for f in sorted(os.listdir(d)):
            if f.startswith("pred_") and f.endswith(".npy") and not f.endswith("_meta.npy"):
                model_name = f[5:-4]
                prob_array = read_array(f"pred_{model_name}", d)
                meta = read_array(f"pred_{model_name}_meta", d, allow_pickle=True, mmap_mode=None)
                level, classes = int(meta[0]), list(meta[1:])
                schema = [f"{model_name}|{cls}" for cls in classes]
                frames.append(pl.DataFrame(prob_array, schema=schema))
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="horizontal")

    def get_features(self) -> pl.DataFrame:
        """Return training feature arrays as a polars DataFrame.

        Requires keep_features=True.
        Returns a DataFrame with columns named by feature type and index,
        e.g. 'quant_r0|0', 'multirocket_r0|1', etc.
        """
        d = self._get_training_dir()
        frames = []
        for f in sorted(os.listdir(d)):
            if f.startswith("Xt_") and f.endswith(".npy") and f != "Xt_probabilities.npy":
                key = f[3:-4]  # e.g. "quant_r_0", "multirocket_r_1"
                arr = read_array(f[:-4], d)
                schema = [f"{key}|{i}" for i in range(arr.shape[1])]
                frames.append(pl.DataFrame(arr, schema=schema))
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="horizontal")

    def summary(self) -> list[dict]:
        """Return OOF scores collected during fit.

        Each entry is a dict with keys: model, level, oof_accuracy.
        Level 0 = base models, level 1 = stacking models.
        """
        return self._oof_scores

    def compute_features(self, X: np.ndarray) -> dict[int, dict[str, np.ndarray]]:
        """Compute features for prediction, returning per-repetition feature dicts.

        Returns {repetition: {feature_type: np.ndarray}}.
        """
        rep_features = {}
        # Quant is computed once at rep 0
        compute_start_time = perf_counter()
        quant_transform = read_model("transformer_quant", self._model_dir, repetition=0)
        X_t = quant_transform.transform(X)
        size_mb = X_t.nbytes / (1024 * 1024)
        shape = X_t.shape
        rep_features[0] = {"quant": X_t}
        duration = perf_counter() - compute_start_time
        self.log(f"Computed QUANT features {shape} ({size_mb:.2f} MB) in {duration:4f}s", level=1, start_time=compute_start_time)
        # Other feature types are per-repetition
        for repetition in range(self.n_repetitions):
            if repetition not in rep_features:
                rep_features[repetition] = {}
            for feature_type in ("MultiRocket", "Hydra", "RDST"):
                feature_start_time = perf_counter()
                transform = read_model(f"transformer_{feature_type.lower()}", self._model_dir, repetition=repetition)
                X_t = transform.transform(X)
                size_mb = X_t.nbytes / (1024 * 1024)
                shape = X_t.shape
                rep_features[repetition][feature_type.lower()] = X_t
                duration = perf_counter() - feature_start_time
                self.log(f"Computed repetition {repetition} {feature_type} features {shape[0],shape[1]} ({size_mb:.2f} MB) in {duration:4f}s", level=1, start_time=compute_start_time)

        return rep_features

    def predict_proba_per_model(self, X):
        import shutil
        predict_start_time = perf_counter()
        self.log("Starting prediction", level=1, start_time=predict_start_time)

        # Create features_inference directory for mmap files
        self._tmpdir = os.path.join(self._base_dir, "features_inference")
        os.makedirs(self._tmpdir, exist_ok=True)
        self.log(f"Starting executor with {self.n_jobs} workers, run_dir={self._base_dir}", level=1, start_time=predict_start_time)


        try:
            with ProcessPoolExecutor(
                    max_workers=self.n_jobs,
                    mp_context=multiprocessing.get_context('spawn'),
            ) as executor:
                futures = [executor.submit(_noop) for _ in range(self.n_jobs)]

                rep_features = self.compute_features(X)
                self.log(f"Computed features for prediction", level=1,
                         start_time=predict_start_time)

                # Save X and per-repetition feature arrays
                save_array(X, "X", self._tmpdir)

                # Quant is computed once (rep 0) and shared across all repetitions
                quant_rep = None
                for rep, feature_dict in rep_features.items():
                    for feat_type, arr in feature_dict.items():
                        save_array(arr, f"Xt_{feat_type}", self._tmpdir, repetition=rep)
                        if feat_type == "quant":
                            quant_rep = rep
                self.log(f"Feature arrays saved to mmap files", level=2,
                         start_time=predict_start_time)

                predictions = []

                # Build tasks from known model structure
                tasks = []
                for model_name in reversed(self.feature_models + self.series_models):
                    is_series = model_name in self.series_models
                    for rep in range(self.n_repetitions):
                        tagged_name = f"{model_name}_r{rep}"
                        feature_specs = {ft: rep for ft in ("quant", "multirocket", "hydra", "rdst")}
                        feature_specs["quant"] = quant_rep  # always shared from first rep
                        for fold in range(self.k_folds):
                            tasks.append((tagged_name, model_name, is_series,
                                          self._tmpdir, feature_specs, self._model_dir, rep, fold))


                self.log(f"Starting prediction with {self.n_jobs} workers for {len(tasks)} first-level models", level=1, start_time=predict_start_time)
                for f in futures:
                    f.result()

                futures = {
                    executor.submit(_predict_one_model_v7, *task): task
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
                    self.log(f"Predicted {model_name} in {predict_dur:.4f}s", level=2, start_time=predict_start_time)

                    level = 0
                    pred_list = self.add_probabilities(proba, classes_, model_name, level)
                    predictions.extend(pred_list)

                self.log(f"Completed all first-level model predictions", level=1, start_time=predict_start_time)

                # Build probability array from level-0 predictions for stacking
                if self._tmpdir and os.path.exists(self._tmpdir):
                    shutil.rmtree(self._tmpdir)
                self._tmpdir = os.path.join(self._base_dir, "features")
                os.makedirs(self._tmpdir, exist_ok=True)

                # Pivot level-0 predictions into probability array (average across folds)
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

                save_array(X, "X", self._tmpdir)
                save_array(prob_array, "Xt_probabilities", self._tmpdir)
                stacking_specs = {"probabilities": None}

                # Build tasks for stacking models
                n_stacking_folds = self.n_repetitions * self.k_folds
                tasks = []
                for model_name in self.stacking_models:
                    is_series = model_name in self.series_models
                    for fold in range(n_stacking_folds):
                        tasks.append((model_name, model_name, is_series,
                                      self._tmpdir, stacking_specs, self._model_dir, 0, fold))

                self.log(f"Starting prediction with {self.n_jobs} workers for {len(tasks)} stacking models",
                         level=1, start_time=predict_start_time)

                futures = {
                    executor.submit(_predict_one_model_v7, *task): task
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
                    self.log(f"Predicted {model_name} in {predict_dur:.4f}s", level=2, start_time=predict_start_time)

                    level = 1
                    pred_list = self.add_probabilities(proba, classes_, model_name, level)
                    predictions.extend(pred_list)

            self.log(f"Completed all stacking model predictions", level=1, start_time=predict_start_time)
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
            all_model_names = []
            for model_name in self.feature_models + self.series_models:
                for rep in range(self.n_repetitions):
                    all_model_names.append(f"{model_name}_r{rep}")
            all_model_names.extend(self.stacking_models)
            for model_name in all_model_names:
                prob_columns = sorted(col for col in all_preds_df.columns if model_name in col)
                agg_probs = all_preds_df.select(prob_columns)
                return_dict[model_name] = agg_probs.to_numpy()

            return return_dict

        finally:
            # Clean up temp directory
            if self._tmpdir and os.path.exists(self._tmpdir):
                shutil.rmtree(self._tmpdir)
                self._tmpdir = None
            self.log("Executor shutdown complete", level=1, start_time=predict_start_time)

    def predict_per_model(self, X):
        """Return hard predictions for each sub-model.

        Returns dict {model_name: np.ndarray of predicted class labels}.
        """
        proba_per_model = self.predict_proba_per_model(X)
        return {
            model_name: self.classes_[np.argmax(proba, axis=1)]
            for model_name, proba in proba_per_model.items()
        }

class LokyStackerV8Base(LokyStackerV7):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, n_jobs=1, keep_features=False, verbose=0):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, n_jobs=n_jobs, keep_features=keep_features, verbose=verbose)

        self.feature_models = ["multirockethydra-bestk-p-ridgecv", "quant-etc", "rdst-p-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []

        stacking_model = "probability-ridgecv"
        # other_stacking_models = ["probability-et", "probability-rf"]
        self.stacking_models = [stacking_model] # + other_stacking_models
        self.best_model = stacking_model

class LokyStackerV8AutoBestStacking(LokyStackerV8Base):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, n_jobs=1, keep_features=False, verbose=0):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, n_jobs=n_jobs, keep_features=keep_features, verbose=verbose)
        self.best_model = "auto-best-stacking"

class LokyStackerV8AutoBestBase(LokyStackerV8Base):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, n_jobs=1, keep_features=False, verbose=0):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, n_jobs=n_jobs, keep_features=keep_features, verbose=verbose)
        self.best_model = "auto-best-base"

class LokyStackerV8AutoBest(LokyStackerV8Base):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, n_jobs=1, keep_features=False, verbose=0):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, n_jobs=n_jobs, keep_features=keep_features, verbose=verbose)
        self.best_model = "auto-best"

class LokyStackerV7SoftFilterRidge(LokyStackerV7):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, n_jobs=1, keep_features=False, verbose=0):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, n_jobs=n_jobs, keep_features=keep_features, verbose=verbose)

        self.feature_models = ["multirockethydra-bestk-p-ridgecv", "quant-etc", "rdst-p-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []

        stacking_model = "probability-ridgecv"
        self.stacking_models = [stacking_model]
        self.best_model = stacking_model


def _make_filter_variant(cls_name, feature_models, series_models):
    """Factory for LokyStackerV7 ablation variants with a specific sub-model combination."""
    _fm = list(feature_models)
    _sm = list(series_models)
    _stacking = "probability-ridgecv"

    class _Variant(LokyStackerV7):
        def __init__(self, random_state=None, n_repetitions=1, k_folds=10,
                     n_jobs=1, keep_features=False, verbose=0):
            super().__init__(
                random_state=random_state, n_repetitions=n_repetitions,
                k_folds=k_folds, n_jobs=n_jobs, keep_features=keep_features,
                verbose=verbose,
            )
            self.feature_models = _fm[:]
            self.series_models = _sm[:]
            self.oof_models = []
            self.stacking_models = [_stacking]
            self.best_model = _stacking

    _Variant.__name__ = cls_name
    _Variant.__qualname__ = cls_name
    return _Variant


_MRH = "multirockethydra-bestk-p-ridgecv"
_Q   = "quant-etc"
_R   = "rdst-p-ridgecv"
_S   = "rstsf"

# Single-component
LokyStackerV7Filter_M    = _make_filter_variant("LokyStackerV7Filter_M",    [_MRH],          [])
LokyStackerV7Filter_Q    = _make_filter_variant("LokyStackerV7Filter_Q",    [_Q],            [])
LokyStackerV7Filter_R    = _make_filter_variant("LokyStackerV7Filter_R",    [_R],            [])
LokyStackerV7Filter_S    = _make_filter_variant("LokyStackerV7Filter_S",    [],              [_S])
# Two-component
LokyStackerV7Filter_MQ   = _make_filter_variant("LokyStackerV7Filter_MQ",   [_MRH, _Q],      [])
LokyStackerV7Filter_MR   = _make_filter_variant("LokyStackerV7Filter_MR",   [_MRH, _R],      [])
LokyStackerV7Filter_MS   = _make_filter_variant("LokyStackerV7Filter_MS",   [_MRH],          [_S])
LokyStackerV7Filter_QR   = _make_filter_variant("LokyStackerV7Filter_QR",   [_Q, _R],        [])
LokyStackerV7Filter_QS   = _make_filter_variant("LokyStackerV7Filter_QS",   [_Q],            [_S])
LokyStackerV7Filter_RS   = _make_filter_variant("LokyStackerV7Filter_RS",   [_R],            [_S])
# Three-component
LokyStackerV7Filter_MQR  = _make_filter_variant("LokyStackerV7Filter_MQR",  [_MRH, _Q, _R],  [])
LokyStackerV7Filter_MQS  = _make_filter_variant("LokyStackerV7Filter_MQS",  [_MRH, _Q],      [_S])
LokyStackerV7Filter_MRS  = _make_filter_variant("LokyStackerV7Filter_MRS",  [_MRH, _R],      [_S])
LokyStackerV7Filter_QRS  = _make_filter_variant("LokyStackerV7Filter_QRS",  [_Q, _R],        [_S])
# All four (equivalent to LokyStackerV7SoftFilterRidge)
LokyStackerV7Filter_MQRS = _make_filter_variant("LokyStackerV7Filter_MQRS", [_MRH, _Q, _R],  [_S])


class LokyStackerV7SoftET(LokyStackerV7):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, n_jobs=1, keep_features=False, verbose=0):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, n_jobs=n_jobs, keep_features=keep_features, verbose=verbose)

        self.feature_models = ["multirockethydra-p-ridgecv", "quant-etc", "rdst-p-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []

        stacking_model = "probability-et"
        self.stacking_models = [stacking_model]
        self.best_model = stacking_model


class LokyStackerV7SoftRidge(LokyStackerV7):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, n_jobs=1, keep_features=False, verbose=0):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, n_jobs=n_jobs, keep_features=keep_features, verbose=verbose)

        self.feature_models = ["multirockethydra-p-ridgecv", "quant-etc", "rdst-p-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []

        stacking_model = "probability-ridgecv"
        self.stacking_models = [stacking_model]
        self.best_model = stacking_model


class LokyStackerV7SoftRF(LokyStackerV7):
    def __init__(self, random_state=None, n_repetitions=1, k_folds=10, n_jobs=1, keep_features=False, verbose=0):
        super().__init__(random_state=random_state, n_repetitions=n_repetitions, k_folds=k_folds, n_jobs=n_jobs, keep_features=keep_features, verbose=verbose)

        self.feature_models = ["multirockethydra-p-ridgecv", "quant-etc", "rdst-p-ridgecv"]
        self.series_models = ["rstsf"]
        self.oof_models = []

        stacking_model = "probability-rf"
        self.stacking_models = [stacking_model]
        self.best_model = stacking_model


def generate_folds(X, y, n_splits=5, n_repetitions=5, random_state=0):
    all_folds = []
    for i in range(n_repetitions):
        folds = utils.get_folds(X, y, n_splits=n_splits, random_state=random_state + i)
        all_folds.extend(folds)
    return all_folds

class TSCGlue(LokyStackerV7SoftFilterRidge):
    def __init__(self, random_state=None, k_folds=10, n_jobs=1, keep_features=False, verbose=0):
        super().__init__(random_state=random_state, n_repetitions=1, k_folds=k_folds, n_jobs=n_jobs, keep_features=keep_features, verbose=verbose)


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
from sklearn.base import BaseEstimator, TransformerMixin
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


class DictMultiScaler(BaseEstimator, TransformerMixin):
    """
    Like MultiScaler but receives a dict of numpy arrays keyed by feature group name.

    Parameters
    ----------
    scalers : dict
        Maps feature group name to scaler instance.
        Example: {'hydra': SparseScaler(), 'multirocket': StandardScaler()}
    """

    def __init__(self, scalers):
        self.scalers = scalers

    def fit(self, X: dict[str, np.ndarray], y=None, idx=None):
        self.scalers_ = {}
        for key, scaler in self.scalers.items():
            if key in X:
                self.scalers_[key] = scaler
                if idx is not None:
                    scaler.fit(X[key][idx])
                else:
                    scaler.fit(X[key])
        return self

    def transform(self, X: dict[str, np.ndarray], idx=None):
        parts = []
        for key in self.scalers_:
            if key in X:
                if idx is not None:
                    parts.append(self.scalers_[key].transform(X[key][idx]))
                else:
                    parts.append(self.scalers_[key].transform(X[key]))
        return np.hstack(parts) if parts else np.empty((next(iter(X.values())).shape[0], 0))

    def fit_transform(self, X: dict[str, np.ndarray], y=None, idx=None):
        return self.fit(X, y, idx=idx).transform(X, idx=idx)


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

from sklearn.utils.extmath import softmax

class RidgeClassifierCVDecisionProba(RidgeClassifierCV):
    def fit(self, X, y):
        with threadpool_limits(limits=1):
            return super().fit(X, y)

    def predict_proba(self, X):
        scores = self.decision_function(X)

        # binary case: decision_function returns shape (n_samples,)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T

        return softmax(scores)


class RSTSFUnsupervisedTransformer:
    def __init__(self, n_intervals=2500, random_state=None, n_jobs=1):
        self.n_intervals = n_intervals
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        lags = int(12 * (X.shape[2] / 100.0) ** 0.25)

        self._series_transformers = [
            FunctionTransformer(func=first_order_differences_3d, validate=False),
            PeriodogramTransformer(),
            ARCoefficientTransformer(order=lags, replace_nan=True),
        ]

        transforms = [X] + [t.fit_transform(X) for t in self._series_transformers]

        self._transformers = []
        for t in transforms:
            ri = RandomIntervals(
                n_intervals=self.n_intervals,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                dilation=[1, 2, 4],
            )
            ri.fit(t)
            self._transformers.append(ri)

        return self

    def transform(self, X):
        transforms = [X] + [t.transform(X) for t in self._series_transformers]
        return np.hstack([self._transformers[i].transform(t) for i, t in enumerate(transforms)])

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class RSTSFUnsupervisedClassifier(BaseClassifier):
    def __init__(self, n_intervals=1000, random_state=None, n_jobs=-1, n_estimators=500):
        super().__init__()
        self.n_intervals = n_intervals
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_estimators = n_estimators

    def _fit(self, X, y):
        self.pipeline_ = make_pipeline(
            RSTSFUnsupervisedTransformer(
                n_intervals=self.n_intervals,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            ),
            ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                criterion="entropy",
                class_weight="balanced",
                max_features="sqrt",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        )
        self.pipeline_.fit(X, y)
        return self

    def _predict_proba(self, X):
        return self.pipeline_.predict_proba(X)

    def _predict(self, X):
        return self.pipeline_.predict(X)
