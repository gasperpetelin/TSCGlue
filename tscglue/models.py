import os
import uuid
import pickle
import shutil
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Tuple

import numpy as np
import polars as pl
from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.interval_based import RSTSF
from aeon.transformations.collection import ARCoefficientTransformer, BaseCollectionTransformer, PeriodogramTransformer
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.transformations.collection.interval_based import QUANTTransformer, RandomIntervals
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from aeon.utils.numba.general import first_order_differences_3d
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.extmath import softmax
from threadpoolctl import threadpool_limits
from tscglue import utils


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
        # Suppress f_classif "Features are constant" warnings: after CV fold splitting
        # and scaling, some features have near-zero variance that passes VarianceThreshold
        # but f_classif still treats as constant. These features are harmless (never selected).
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
    elif name == "probability-tabicl":
        from tabicl import TabICLClassifier
        scaler = DictMultiScaler(scalers={"probabilities": NoScaler()})
        clf = TabICLClassifier(device="cpu", random_state=seed, n_jobs=1, kv_cache=False)
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
    elif name == "fm-dummy":
        from sklearn.dummy import DummyClassifier
        scaler = DictMultiScaler(scalers={"mantis": NoScaler(), "chronos2": NoScaler()})
        clf = DummyClassifier(strategy="prior")
        return scaler, clf
    elif name == "fm-p-ridgecv":
        scaler = DictMultiScaler(scalers={"mantis": StandardScaler(), "chronos2": StandardScaler()})
        clf = RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10))
        return scaler, clf
    elif name == "tsfresh-rotf":
        from aeon.classification.sklearn import RotationForestClassifier
        scaler = DictMultiScaler(scalers={"tsfresh": NoScaler()})
        clf = RotationForestClassifier(n_estimators=200, n_jobs=n_jobs, random_state=seed)
        return scaler, clf
    elif name == "rstsf":
        return None, RSTSF(random_state=seed, n_jobs=n_jobs, n_estimators=100)
    elif name == "rstsf-combined-etc":
        scaler = DictMultiScaler(scalers={"rstsf-combined": NoScaler()})
        clf = ExtraTreesClassifier(
            n_estimators=200, criterion="entropy", class_weight="balanced",
            max_features="sqrt", n_jobs=n_jobs, random_state=seed,
        )
        return scaler, clf
    elif name == "rstsf-random-etc":
        scaler = DictMultiScaler(scalers={"rstsf-random": NoScaler()})
        clf = ExtraTreesClassifier(
            n_estimators=200, criterion="entropy", class_weight="balanced",
            max_features="sqrt", n_jobs=n_jobs, random_state=seed,
        )
        return scaler, clf
    else:
        raise ValueError(f"Unknown model name: {name}")




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
    directory = str(directory)
    rep_suffix = f"_r_{repetition}" if repetition is not None else ""
    fold_suffix = f"_f_{fold}" if fold is not None else ""
    path = f"{directory}/{name}{rep_suffix}{fold_suffix}.pkl"
    with open(path, 'rb') as f:
        return pickle.load(f)

class RDSTFloat64(RandomDilatedShapeletTransform):
    """RDST wrapper that casts input to float64 (numba requires it)."""

    def _fit(self, X, y=None):
        return super()._fit(np.asarray(X, dtype=np.float64), y)

    def _transform(self, X, y=None):
        return super()._transform(np.asarray(X, dtype=np.float64), y)


def get_feature_transformer(feature_type: str, seed: int, n_jobs: int = 1):
    match feature_type:
        case "multirocket":
            return MultiRocket(n_jobs=n_jobs, random_state=seed)
        case "rdst":
            return RDSTFloat64(n_jobs=n_jobs, random_state=seed)
        case "quant":
            return QUANTTransformer()
        case "hydra":
            return HydraTransformer(n_jobs=n_jobs, random_state=seed)
        case "mantis":
            from tscglue.models_tsfm import MantisEmbedding
            return MantisEmbedding()
        case "chronos2":
            from tscglue.models_tsfm import Chronos2Embedding
            return Chronos2Embedding()
        case "tsfresh":
            from aeon.transformations.collection.feature_based import TSFresh
            return TSFresh(default_fc_parameters="efficient", n_jobs=n_jobs)
        case "rstsf-combined":
            from tscglue.interval_models import RSTSFCombinedTransformer
            return RSTSFCombinedTransformer(n_jobs=n_jobs, random_state=seed)
        case "rstsf-random":
            from tscglue.interval_models import RSTSFRandomTransformer
            return RSTSFRandomTransformer(n_jobs=n_jobs, random_state=seed)
        case _:
            raise ValueError(f"Unknown feature transformer type: {feature_type}")

def _noop():
    return None

def _run_in_subprocess(target, args):
    """Run target(*args) in a fresh spawned subprocess.

    Uses 'spawn' context to avoid inheriting state (e.g. GPU memory, open file
    descriptors, TensorFlow/PyTorch globals) from the parent process, which can
    cause hangs or incorrect behaviour with foundation-model transformers.
    """
    mp_ctx = multiprocessing.get_context("forkserver")
    p = mp_ctx.Process(target=target, args=args)
    p.start()
    p.join()
    if p.exitcode != 0:
        raise RuntimeError(f"Subprocess failed with exit code {p.exitcode}")

def _fit_transformer_in_subprocess(feature_name, feature_seed, n_jobs, X_path, model_dir, feature_id):
    X = np.load(X_path, allow_pickle=True)
    transformer = get_feature_transformer(feature_name, seed=feature_seed, n_jobs=n_jobs)
    transformer.fit(X)
    save_model(transformer, f"transformer_{feature_id}", model_dir)

def _transform_in_subprocess(feature_id, X_path, model_dir, output_dir, dtype=np.float64):
    X = np.load(X_path, allow_pickle=True)
    transformer = read_model(f"transformer_{feature_id}", model_dir)
    t0 = perf_counter()
    Xt = transformer.transform(X)
    print(f"[subprocess] transform {feature_id}: {perf_counter() - t0:.4f}s")
    save_array(Xt, f"Xt_{feature_id}", output_dir, dtype=dtype)

def _fit_transform_in_subprocess(feature_name, feature_seed, n_jobs, X_path, model_dir, output_dir, feature_id, dtype=np.float64):
    X = np.load(X_path, allow_pickle=True)
    transformer = get_feature_transformer(feature_name, seed=feature_seed, n_jobs=n_jobs)
    t0 = perf_counter()
    Xt = transformer.fit_transform(X)
    print(f"[subprocess] fit_transform {feature_id}: {perf_counter() - t0:.4f}s")
    save_model(transformer, f"transformer_{feature_id}", model_dir)
    save_array(Xt, f"Xt_{feature_id}", output_dir, dtype=dtype)

@dataclass(frozen=True)
class FeatureSpec:
    feature_name: str
    feature_seed: int | None = None

    def get_feature_id(self):
        return f'{self.feature_name}_s_{self.feature_seed}' if self.feature_seed is not None else self.feature_name

@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_seed: int
    is_series: bool
    level: int
    features: Tuple[FeatureSpec, ...]
    fold_seeds: tuple[int, ...]

    def get_model_id(self):
        return f'{self.model_name}_s_{self.model_seed}'

    @property
    def n_repetitions(self) -> int:
        return len(self.fold_seeds)

def _load_feature_dict_v10(directory, feature_specs):
    """Load feature arrays using read_array with (feat_type, repetition) specs."""
    feature_dict = {}
    for feat_spec in feature_specs:
        if feat_spec.feature_name == "raw":
            continue
        feat_id = feat_spec.get_feature_id()
        feature_dict[feat_spec.feature_name] = read_array(f"Xt_{feat_id}", directory)
    return feature_dict

def _predict_one_model_v10(model_id, model_name, is_series, directory, feature_specs, model_dir, fold):
    """Prediction function - loads model from disk, loads data via read_array."""
    X = read_array("X", directory)
    feature_dict = _load_feature_dict_v10(directory, feature_specs)

    scaler, clf = read_model(model_id, model_dir, None, fold)
    start_predict = perf_counter()

    if is_series:
        proba = clf.predict_proba(X)
    else:
        X_scaled = scaler.transform(feature_dict)
        proba = clf.predict_proba(X_scaled)

    predict_dur = perf_counter() - start_predict
    return (proba, clf.classes_, predict_dur, model_id)

def _train_one_model_v10(fold_number, model_id, model_name, is_series, train_idx, val_idx, model_seed,
                         directory, feature_specs, model_dir):
    """Training function - loads data via read_array, saves model to disk."""
    X = read_array("X", directory)
    y = read_array("y", directory)
    feature_dict = _load_feature_dict_v10(directory, feature_specs)

    scaler, clf = get_model_v6(model_name, seed=model_seed)
    start_train = perf_counter()

    if is_series:
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[val_idx])
        _, model_size = save_model((None, clf), model_id, model_dir, None, fold_number)
    else:
        clf.fit(scaler.fit_transform(feature_dict, idx=train_idx), y[train_idx])
        proba = clf.predict_proba(scaler.transform(feature_dict, idx=val_idx))
        _, model_size = save_model((scaler, clf), model_id, model_dir, None, fold_number)

    train_dur = perf_counter() - start_train
    return (train_idx, val_idx, proba, clf.classes_, model_size, train_dur, model_id, fold_number)

class LokyStackerV10Base(BaseClassifier):
    _tags = {"capability:multivariate": True}

    DEFAULT_MODEL_NAMES = [
        "multirockethydra-bestk-p-ridgecv", "quant-etc", "rdst-p-ridgecv", "rstsf",
    ]
    SERIES_MODELS = ["rstsf"]
    STACKING_MODEL = "probability-ridgecv"

    def _get_feature_names(self, model_name: str) -> tuple[str, ...]:
        """Return required feature type names for a model."""
        if model_name in ("multirockethydra-bestk-p-ridgecv", "multirockethydra-p-ridgecv", "multirockethydra-ridgecv"):
            return ("multirocket", "hydra")
        elif model_name == "quant-etc":
            return ("quant",)
        elif model_name in ("rdst-p-ridgecv", "rdst-ridgecv"):
            return ("rdst",)
        elif model_name == "rstsf":
            return ("raw",)
        elif model_name == "rstsf-combined-etc":
            return ("rstsf-combined",)
        elif model_name == "rstsf-random-etc":
            return ("rstsf-random",)
        elif model_name in ("fm-dummy", "fm-p-ridgecv"):
            return ("mantis", "chronos2")
        elif model_name == "tsfresh-rotf":
            return ("tsfresh",)
        else:
            raise ValueError(f"Unknown model {model_name}")

    def _make_feature_spec(self, feature_name: str, group_rng: np.random.Generator) -> FeatureSpec:
        """Create a single FeatureSpec. Seedless for deterministic transforms like quant."""
        if feature_name in ("quant", "raw", "mantis", "chronos2", "tsfresh"):
            return FeatureSpec(feature_name=feature_name)
        return FeatureSpec(feature_name=feature_name, feature_seed=int(group_rng.integers(0, 2**31 - 1)))

    def build_model_specs(self, model_names: list[str]) -> list[ModelSpec]:
        """Build ModelSpec list from a flat list of model names.

        Models are accumulated into groups that share feature seeds.
        A duplicate model name starts a new group.
        """
        # Split flat list into groups: a new group starts when a name is repeated
        groups: list[list[str]] = []
        seen: set[str] = set()
        for name in model_names:
            if name in seen:
                groups.append([])
                seen = set()
            if not groups:
                groups.append([])
            groups[-1].append(name)
            seen.add(name)

        all_models: list[ModelSpec] = []
        for group in groups:
            group_rng = np.random.default_rng(self._get_feature_seed())

            # Build FeatureSpecs per group, deduped by feature name within group
            group_features: dict[str, FeatureSpec] = {}
            for model_name in group:
                for ft_name in self._get_feature_names(model_name):
                    if ft_name not in group_features:
                        group_features[ft_name] = self._make_feature_spec(ft_name, group_rng)

            for model_name in group:
                is_series = model_name in self.series_models
                features = tuple(
                    group_features[ft_name] for ft_name in self._get_feature_names(model_name)
                )
                model_seed = self._get_feature_seed()
                fold_seed_rng = np.random.default_rng(model_seed)
                fold_seeds = tuple(
                    int(fold_seed_rng.integers(0, 2**31 - 1)) for _ in range(self.n_repetitions)
                )
                spec = ModelSpec(
                    model_name=model_name,
                    model_seed=model_seed,
                    is_series=is_series,
                    level=0,
                    features=features,
                    fold_seeds=fold_seeds,
                )
                all_models.append(spec)

        return all_models

    def __init__(self, random_state=None, k_folds=10, n_jobs=1, keep_features=False, verbose=0,
                 model_names=None, n_repetitions=1, feature_dtype=None):
        super().__init__()
        self.k_folds = int(k_folds)
        self.random_state = random_state
        self.n_jobs = int(n_jobs)
        self.keep_features = bool(keep_features)
        self.verbose = int(verbose)
        self.n_repetitions = int(n_repetitions)
        self.feature_dtype = np.dtype(feature_dtype) if feature_dtype is not None else None

        self.cv_splits = None
        self.feature_seed = np.random.default_rng(random_state)

        self._run_id = uuid.uuid4().hex[:16]
        self._base_dir = Path(".", "tscglue_runs", self._run_id)
        self._model_dir = self._base_dir / "models"
        self._tmpdir: Path | None = self._base_dir / "features_training"

        self.model_names = model_names
        self.series_models = self.SERIES_MODELS.copy()
        self.stacking_models = [self.STACKING_MODEL]
        self.best_model = self.STACKING_MODEL

        # Build model specs from flat list; derive unique features
        self.model_specs = self.build_model_specs(
            self.model_names if self.model_names is not None else self.DEFAULT_MODEL_NAMES
        )
        all_features: dict[str, FeatureSpec] = {}
        for spec in self.model_specs:
            for ft in spec.features:
                fid = ft.get_feature_id()
                if fid not in all_features:
                    all_features[fid] = ft
        self.features_list = list(all_features.values())

        self._oof_scores: list[dict] = []
        self._probability_columns: list[tuple[int, str, Any]] | None = None
        self.neki: list[dict] = []

        self._fallback_path: Path = self._model_dir / "fallback.pkl"

    # ----------------- utils -----------------

    def _get_feature_seed(self) -> int:
        return int(self.feature_seed.integers(0, 2**31 - 1, dtype=np.int32))

    def log(self, message: str, level: int, start_time=None, current_time=None):
        if self.verbose >= level:
            if start_time is not None:
                if current_time is None:
                    current_time = perf_counter()
                print(f"[{current_time - start_time:.2f}s] {message}")
            else:
                print(message)

    def _require_tmpdir(self) -> Path:
        if self._tmpdir is None:
            raise RuntimeError("Temporary directory not available.")
        return self._tmpdir

    def _label_to_python(self, value: Any) -> Any:
        return value.item() if isinstance(value, np.generic) else value

    def _label_sort_key(self, value: Any) -> tuple[str, str]:
        value = self._label_to_python(value)
        return type(value).__name__, repr(value)

    def _probability_key(self, level: int, model_name: str, cls: Any) -> tuple[int, str, Any]:
        return int(level), model_name, self._label_to_python(cls)

    def _probability_sort_key(self, key: tuple[int, str, Any]) -> tuple[int, str, tuple[str, str]]:
        level, model_name, cls = key
        return level, model_name, self._label_sort_key(cls)

    def _aggregate_prediction_matrix(
        self,
        predictions: list[dict],
        n_samples: int,
        probability_columns: Iterable[tuple[int, str, Any]],
    ) -> np.ndarray:
        columns = list(probability_columns)
        if not columns:
            return np.empty((n_samples, 0), dtype=np.float64)

        col_to_idx = {col: i for i, col in enumerate(columns)}
        prob_sum = np.zeros((n_samples, len(columns)), dtype=np.float64)
        prob_count = np.zeros((n_samples, len(columns)), dtype=np.int32)

        for pred in predictions:
            key = self._probability_key(pred["level"], pred["model"], pred["class"])
            col_idx = col_to_idx.get(key)
            if col_idx is None:
                continue
            row_idx = int(pred["index"])
            prob_sum[row_idx, col_idx] += float(pred["probability"])
            prob_count[row_idx, col_idx] += 1

        prob_array = np.full((n_samples, len(columns)), np.nan, dtype=np.float64)
        np.divide(prob_sum, prob_count, out=prob_array, where=prob_count > 0)
        return prob_array

    def cleanup(self):
        if self._base_dir.exists():
            shutil.rmtree(self._base_dir)

    def _feature_input(self, ft: str, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float64) if ft == "rdst" else X

    # ----------------- aeon API -----------------

    def _predict_proba(self, X):
        if self._fallback_path.exists():
            fallback = read_model("fallback", str(self._model_dir))
            return fallback.predict_proba(X)
        return self.predict_proba_per_model(X)[self.best_model]

    def _predict(self, X):
        if self._fallback_path.exists():
            fallback = read_model("fallback", str(self._model_dir))
            return fallback.predict(X)
        probas = self._predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

    # ----------------- prediction row helpers -----------------

    def add_probabilities(self, probas, classes, model_name, level, indices=None):
        preds = []
        row_indices = np.arange(len(probas)) if indices is None else np.asarray(indices)
        for idx, row in zip(row_indices, probas):
            for scls, prob in zip(classes, row):
                preds.append(
                    {
                        "index": int(idx),
                        "model": model_name,
                        "level": level,
                        "class": self._label_to_python(scls),
                        "probability": float(self._label_to_python(prob)),
                    }
                )
        return preds

    # ----------------- OOF persistence -----------------

    def _save_model_predictions(self, predictions, model_name, n_samples, level):
        model_preds = [p for p in predictions if p["model"] == model_name]
        if not model_preds:
            return predictions
        classes = sorted({self._label_to_python(p["class"]) for p in model_preds}, key=self._label_sort_key)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        prob_sum = np.zeros((n_samples, len(classes)), dtype=np.float64)
        prob_count = np.zeros((n_samples, len(classes)), dtype=np.int32)
        for p in model_preds:
            idx = p["index"]
            cidx = class_to_idx[p["class"]]
            prob_sum[idx, cidx] += p["probability"]
            prob_count[idx, cidx] += 1
        prob_array = np.where(prob_count > 0, prob_sum / prob_count, np.nan)
        d = str(self._require_tmpdir())
        save_array(prob_array, f"pred_{model_name}", d)
        save_array(np.array([level] + classes, dtype=object), f"pred_{model_name}_meta", d)
        return [p for p in predictions if p["model"] != model_name]

    def _load_model_predictions(self, model_name):
        d = str(self._require_tmpdir())
        prob_array = read_array(f"pred_{model_name}", d)
        meta = read_array(f"pred_{model_name}_meta", d, allow_pickle=True, mmap_mode=None)
        level = int(meta[0])
        classes = list(meta[1:])
        return prob_array, level, classes

    def _compute_oof_accuracy(self, y, model_name) -> float:
        prob_array, _level, classes = self._load_model_predictions(model_name)
        valid = ~np.isnan(prob_array).any(axis=1)
        if not np.any(valid):
            return 0.0
        pred_idx = np.argmax(prob_array[valid], axis=1)
        preds = np.asarray(classes)[pred_idx]
        y_true = y[np.where(valid)[0]]
        y_true_str = np.asarray(y_true, dtype=str)
        preds_str = np.asarray(preds, dtype=str)
        return float(accuracy_score(y_true_str, preds_str))

    def _build_probability_array(self, n_samples: int):
        d = self._require_tmpdir()
        prob_files = sorted(p for p in d.glob("pred_*.npy") if not p.name.endswith("_meta.npy"))
        cols, names = [], []
        for path in prob_files:
            model_name = path.stem[5:]  # strip pred_
            prob_array, level, classes = self._load_model_predictions(model_name)
            if level != 0:
                continue
            for i, cls in enumerate(classes):
                names.append(self._probability_key(level, model_name, cls))
                cols.append(prob_array[:, i])
        if not cols:
            return None
        order = sorted(range(len(names)), key=lambda i: self._probability_sort_key(names[i]))
        self._probability_columns = [names[i] for i in order]
        return np.column_stack([cols[i] for i in order])

    # ----------------- features: train transformers + compute arrays -----------------

    def fit_transform_features(self, X: np.ndarray, fit_start_time=None) -> None:
        """Fit transformers and compute features in a single subprocess per transformer."""
        os.makedirs(self._model_dir, exist_ok=True)
        directory = str(self._tmpdir)
        X_path = str(self._tmpdir / "X.npy")
        for ft in self.features_list:
            if ft.feature_name == "raw":
                continue
            t0 = perf_counter()
            _run_in_subprocess(
                _fit_transform_in_subprocess,
                (ft.feature_name, ft.feature_seed, self.n_jobs,
                 X_path, str(self._model_dir), directory, ft.get_feature_id(), self.feature_dtype),
            )
            Xt = read_array(f"Xt_{ft.get_feature_id()}", directory)
            size_mb = Xt.nbytes / (1024 * 1024)
            self.log(
                f"Fit+transformed {ft.get_feature_id()} features {Xt.shape} ({size_mb:.2f} MB) dtype={Xt.dtype} in {perf_counter() - t0:.4f}s",
                level=1,
                start_time=fit_start_time,
            )

    def compute_features(self, X: np.ndarray, directory: str, start_time=None) -> None:
        compute_start = perf_counter()
        X_path = f"{directory}/X.npy"
        for ft in self.features_list:
            if ft.feature_name == "raw":
                continue
            t0 = perf_counter()
            _run_in_subprocess(
                _transform_in_subprocess,
                (ft.get_feature_id(), X_path, str(self._model_dir), directory, self.feature_dtype),
            )
            Xt = read_array(f"Xt_{ft.get_feature_id()}", directory)
            size_mb = Xt.nbytes / (1024 * 1024)
            self.log(
                f"Computed {ft.get_feature_id()} features {Xt.shape} ({size_mb:.2f} MB) dtype={Xt.dtype} in {perf_counter() - t0:.4f}s",
                level=1,
                start_time=compute_start if start_time is None else start_time,
            )

    # ----------------- fallback -----------------

    def _fit_fallback(self, X, y, fit_start_time):
        self.log("Falling back to MultiRocketHydraClassifier", level=1, start_time=fit_start_time)
        fallback = MultiRocketHydraClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        fallback.fit(X, y)
        save_model(fallback, "fallback", str(self._model_dir))
        self.log("Fallback model trained successfully", level=1, start_time=fit_start_time)

    # ----------------- training -----------------

    def _fit(self, X, y):
        fit_start = perf_counter()
        if self.feature_dtype is None:
            self.feature_dtype = np.asarray(X).dtype
        self.log(f"Starting fit, run_dir={self._base_dir}, n_jobs={self.n_jobs}", level=1, start_time=fit_start)

        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._tmpdir, exist_ok=True)

        t0 = perf_counter()
        save_array(X, "X", str(self._tmpdir), dtype=self.feature_dtype)
        save_array(y, "y", str(self._tmpdir))
        self.log(f"Saved X and y to disk in {perf_counter() - t0:.2f}s (dtype={self.feature_dtype})", level=2, start_time=fit_start)

        _, counts = np.unique(y, return_counts=True)
        if np.any(counts < 2):
            self.log("Some classes have fewer than 2 instances, fold training not possible", level=1, start_time=fit_start)
            self._fit_fallback(X, y, fit_start)
            return

        if self.cv_splits is None:
            self.cv_splits = []

        mp_ctx = multiprocessing.get_context("forkserver")

        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=mp_ctx) as executor:
                warm = [executor.submit(_noop) for _ in range(self.n_jobs)]
                predictions = []

                self.fit_transform_features(X, fit_start_time=fit_start)
                stacker_fold_seed = self._get_feature_seed()

                # -------- level 0 --------
                expected_folds = {spec.get_model_id(): self.k_folds * spec.n_repetitions for spec in self.model_specs}
                tasks = []
                for spec in self.model_specs:
                    fold_rng = np.random.default_rng(spec.model_seed)
                    fold_counter = 0
                    for fold_seed in spec.fold_seeds:
                        rep_splits = generate_folds(X, y, n_splits=self.k_folds, n_repetitions=1, random_state=fold_seed)
                        for _, (train_idx, val_idx) in enumerate(rep_splits):
                            fold_model_seed = int(fold_rng.integers(0, 2**31 - 1))
                            tasks.append(
                                (
                                    fold_counter,
                                    spec.get_model_id(),
                                    spec.model_name,
                                    spec.is_series,
                                    train_idx,
                                    val_idx,
                                    fold_model_seed,
                                    str(self._tmpdir),
                                    list(spec.features),
                                    str(self._model_dir),
                                )
                            )
                            fold_counter += 1

                n_workers = min(self.n_jobs, len(tasks))
                self.log(f"Starting training with {n_workers} workers for {len(tasks)} models", level=2, start_time=fit_start)

                futures = {executor.submit(_train_one_model_v10, *t): t for t in tasks}
                model_groups = defaultdict(list)

                for future in as_completed(futures):
                    task = futures[future]
                    fold_number = task[0]
                    model_id_task = task[1]
                    try:
                        train_idx, val_idx, proba, classes_, model_size, train_dur, model_id_result, fold_number = future.result()
                    except Exception as e:
                        raise RuntimeError(f"Worker failed during training {model_id_task} fold {fold_number}: {e}") from e

                    self.log(
                        f"Trained {model_id_result} in {train_dur:.4f}s for f-{fold_number} "
                        f"({model_size / (1024 * 1024):.2f} MB)",
                        level=2,
                        start_time=fit_start,
                    )

                    new_preds = self.add_probabilities(
                        probas=proba,
                        classes=classes_,
                        model_name=model_id_result,
                        level=0,
                        indices=val_idx,
                    )
                    predictions.extend(new_preds)
                    self.neki.extend(new_preds)

                    model_groups[model_id_result].append(fold_number)
                    if len(model_groups[model_id_result]) == expected_folds[model_id_result]:
                        self.log(f"Completed training for model {model_id_result}", level=2, start_time=fit_start)
                        del model_groups[model_id_result]

                        predictions = self._save_model_predictions(predictions, model_id_result, n_samples=X.shape[0], level=0)
                        oof_acc = self._compute_oof_accuracy(y, model_id_result)
                        self._oof_scores.append({"model": model_id_result, "level": 0, "oof_accuracy": oof_acc})
                        self.log(f"OOF acc (base) {model_id_result}: {oof_acc}", level=1, start_time=fit_start)

                # -------- stacking --------
                self.log("Starting stacking model training", level=2, start_time=fit_start)
                prob_array = self._build_probability_array(n_samples=X.shape[0])
                if prob_array is None or np.isnan(prob_array).any():
                    self.log("NaN values detected in probability array, skipping stacking", level=2, start_time=fit_start)
                    self._fit_fallback(X, y, fit_start)
                    return

                save_array(prob_array, "Xt_probabilities", str(self._tmpdir))

                stacker_splits = generate_folds(X, y, n_splits=self.k_folds, n_repetitions=1, random_state=stacker_fold_seed)
                stack_tasks = []
                for model_name in self.stacking_models:
                    stack_fold_rng = np.random.default_rng(self._get_feature_seed())
                    for fold_no, (train_idx, val_idx) in enumerate(stacker_splits):
                        stack_fold_seed = int(stack_fold_rng.integers(0, 2**31 - 1))
                        stack_tasks.append(
                            (
                                fold_no,
                                model_name,  # model_id = model_name for stacking
                                model_name,
                                False,
                                train_idx,
                                val_idx,
                                stack_fold_seed,
                                str(self._tmpdir),
                                [FeatureSpec(feature_name="probabilities")],
                                str(self._model_dir),
                            )
                        )

                n_workers = min(self.n_jobs, len(stack_tasks))
                self.log(f"Starting stacking training with {n_workers} workers for {len(stack_tasks)} models", level=2, start_time=fit_start)

                futures = {executor.submit(_train_one_model_v10, *t): t for t in stack_tasks}
                model_groups = defaultdict(list)

                for future in as_completed(futures):
                    task = futures[future]
                    fold_number = task[0]
                    model_id_task = task[1]
                    try:
                        train_idx, val_idx, proba, classes_, model_size, train_dur, model_id_result, fold_number = future.result()
                    except Exception as e:
                        raise RuntimeError(f"Worker failed during stacking training {model_id_task} fold {fold_number}: {e}") from e

                    self.log(
                        f"Trained {model_id_result} in {train_dur:.4f}s for f-{fold_number} "
                        f"({model_size / (1024 * 1024):.2f} MB)",
                        level=2,
                        start_time=fit_start,
                    )

                    new_preds = self.add_probabilities(
                        probas=proba,
                        classes=classes_,
                        model_name=model_id_result,
                        level=1,
                        indices=val_idx,
                    )
                    predictions.extend(new_preds)
                    self.neki.extend(new_preds)

                    model_groups[model_id_result].append(fold_number)
                    if len(model_groups[model_id_result]) == self.k_folds:
                        self.log(f"Completed training for model {model_id_result}", level=2, start_time=fit_start)
                        del model_groups[model_id_result]

                        predictions = self._save_model_predictions(predictions, model_id_result, n_samples=X.shape[0], level=1)
                        oof_acc = self._compute_oof_accuracy(y, model_id_result)
                        self._oof_scores.append({"model": model_id_result, "level": 1, "oof_accuracy": oof_acc})
                        self.log(f"OOF acc (stack) {model_id_result}: {oof_acc}", level=1, start_time=fit_start)

                self.log("Fit complete", level=1, start_time=fit_start)

        finally:
            if not self.keep_features and self._tmpdir and self._tmpdir.exists():
                cleanup_start = perf_counter()
                shutil.rmtree(self._tmpdir)
                self.log(f"Cleaned up tmpdir in {perf_counter() - cleanup_start:.2f}s", level=2, start_time=fit_start)
                self._tmpdir = None
            if self.keep_features and self._tmpdir:
                self.features_training_dir_ = str(self._tmpdir)
            self.log("Executor shutdown complete", level=2, start_time=fit_start)

    # ----------------- inspection helpers -----------------

    def _get_training_dir(self) -> str:
        d = getattr(self, "features_training_dir_", None) or (str(self._tmpdir) if self._tmpdir else None)
        if not self.keep_features or not d or not os.path.exists(d):
            raise RuntimeError(
                f"Not available. Set keep_features=True before fitting. keep_features={self.keep_features}, dir={d}"
            )
        return d

    def get_oof_predictions(self) -> pl.DataFrame:
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
        return pl.DataFrame() if not frames else pl.concat(frames, how="horizontal")

    def get_features(self) -> pl.DataFrame:
        d = self._get_training_dir()
        frames = []
        for f in sorted(os.listdir(d)):
            if f.startswith("Xt_") and f.endswith(".npy") and f != "Xt_probabilities.npy":
                key = f[3:-4]
                arr = read_array(f[:-4], d)
                schema = [f"{key}|{i}" for i in range(arr.shape[1])]
                frames.append(pl.DataFrame(arr, schema=schema))
        return pl.DataFrame() if not frames else pl.concat(frames, how="horizontal")

    def summary(self) -> list[dict]:
        return self._oof_scores

    # ----------------- inference -----------------

    def predict_proba_per_model(self, X: np.ndarray) -> dict[str, np.ndarray]:
        predict_start = perf_counter()
        self.log("Starting prediction", level=1, start_time=predict_start)

        mp_ctx = multiprocessing.get_context("forkserver")
        features_infer = self._base_dir / "features_inference"
        features_stack = self._base_dir / "features"

        os.makedirs(features_infer, exist_ok=True)
        self._tmpdir = features_infer

        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=mp_ctx) as executor:
                warm = [executor.submit(_noop) for _ in range(self.n_jobs)]

                # compute features (transform-only; transformers already trained)
                save_array(X, "X", str(features_infer), dtype=self.feature_dtype)
                self.compute_features(X, str(features_infer), start_time=predict_start)
                self.log("Computed and saved features for prediction", level=1, start_time=predict_start)

                predictions = []
                # ---- level 0 predictions ----
                tasks = []
                for spec in self.model_specs:
                    for fold in range(self.k_folds * spec.n_repetitions):
                        tasks.append(
                            (
                                spec.get_model_id(),
                                spec.model_name,
                                spec.is_series,
                                str(features_infer),
                                list(spec.features),
                                str(self._model_dir),
                                fold,
                            )
                        )

                self.log(
                    f"Starting prediction with {self.n_jobs} workers for {len(tasks)} first-level models",
                    level=1,
                    start_time=predict_start,
                )

                futures = {executor.submit(_predict_one_model_v10, *t): t for t in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    model_id_task = task[0]
                    try:
                        proba, classes_, predict_dur, model_id_res = future.result()
                    except Exception as e:
                        raise RuntimeError(f"Worker failed during prediction {model_id_task}: {e}") from e

                    self.log(f"Predicted {model_id_res} in {predict_dur:.4f}s", level=2, start_time=predict_start)
                    predictions.extend(self.add_probabilities(proba, classes_, model_id_res, level=0))

                self.log("Completed all first-level model predictions", level=1, start_time=predict_start)

                # ---- build stacking matrix ----
                if features_infer.exists():
                    shutil.rmtree(features_infer)
                os.makedirs(features_stack, exist_ok=True)
                self._tmpdir = features_stack

                if self._probability_columns is None:
                    raise RuntimeError("Probability column metadata missing. Fit the model before predicting.")
                prob_array = self._aggregate_prediction_matrix(
                    predictions=predictions,
                    n_samples=X.shape[0],
                    probability_columns=self._probability_columns,
                )

                save_array(X, "X", str(features_stack), dtype=self.feature_dtype)
                save_array(prob_array, "Xt_probabilities", str(features_stack))

                # ---- stacking predictions ----
                stack_tasks = []
                for model_name in self.stacking_models:
                    for fold in range(self.k_folds):
                        stack_tasks.append(
                            (
                                model_name,  # model_id = model_name for stacking
                                model_name,
                                False,
                                str(features_stack),
                                [FeatureSpec(feature_name="probabilities")],
                                str(self._model_dir),
                                fold,
                            )
                        )

                self.log(
                    f"Starting prediction with {self.n_jobs} workers for {len(stack_tasks)} stacking models",
                    level=1,
                    start_time=predict_start,
                )

                futures = {executor.submit(_predict_one_model_v10, *t): t for t in stack_tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    model_id_task = task[0]
                    try:
                        proba, classes_, predict_dur, model_id_res = future.result()
                    except Exception as e:
                        raise RuntimeError(f"Worker failed during stacking prediction {model_id_task}: {e}") from e

                    self.log(f"Predicted {model_id_res} in {predict_dur:.4f}s", level=2, start_time=predict_start)
                    predictions.extend(self.add_probabilities(proba, classes_, model_id_res, level=1))

            self.log("Completed all stacking model predictions", level=1, start_time=predict_start)

            model_ids = [spec.get_model_id() for spec in self.model_specs] + self.stacking_models
            out = {}
            for model_id in model_ids:
                level = 1 if model_id in self.stacking_models else 0
                cols = [self._probability_key(level, model_id, cls) for cls in self.classes_]
                out[model_id] = self._aggregate_prediction_matrix(
                    predictions=predictions,
                    n_samples=X.shape[0],
                    probability_columns=cols,
                )
            return out

        finally:
            for d in (features_infer, features_stack):
                if d.exists():
                    shutil.rmtree(d)
            self._tmpdir = None
            self.log("Executor shutdown complete", level=1, start_time=predict_start)

    def predict_per_model(self, X: np.ndarray) -> dict[str, np.ndarray]:
        proba_per_model = self.predict_proba_per_model(X)
        return {name: self.classes_[np.argmax(proba, axis=1)] for name, proba in proba_per_model.items()}


class LokyStackerV10TabICL(LokyStackerV10Base):
    STACKING_MODEL = "probability-tabicl"


class LokyStackerV10FM(LokyStackerV10Base):
    """LokyStackerV10Base + mantis/chronos2 foundation model features with RidgeCV."""

    DEFAULT_MODEL_NAMES = LokyStackerV10Base.DEFAULT_MODEL_NAMES + ["fm-p-ridgecv"]


class LokyStackerV10FMTSFresh(LokyStackerV10FM):
    """LokyStackerV10FM + TSFresh efficient features with RotationForest."""

    DEFAULT_MODEL_NAMES = LokyStackerV10FM.DEFAULT_MODEL_NAMES + ["tsfresh-rotf"]


class LokyStackerV10RSTSFRandom(LokyStackerV10Base):
    """LokyStackerV10FMTSFresh with rstsf replaced by 2-stage rstsf-random-etc.

    Identical to LokyStackerV10FMTSFresh but splits RSTSF into a feature
    extraction transformer (RandomIntervals on 4 series representations) and a
    separately trained ExtraTreesClassifier. SERIES_MODELS is empty so all
    models go through the feature caching pipeline.
    """

    DEFAULT_MODEL_NAMES = [
        "multirockethydra-bestk-p-ridgecv", "quant-etc", "rdst-p-ridgecv",
        "rstsf-random-etc", "fm-p-ridgecv",
    ]
    SERIES_MODELS = []

def generate_folds(X, y, n_splits=5, n_repetitions=5, random_state=0):
    all_folds = []
    for i in range(n_repetitions):
        folds = utils.get_folds(X, y, n_splits=n_splits, random_state=random_state + i)
        all_folds.extend(folds)
    return all_folds

class TSCGlue(LokyStackerV10FM):
    def __init__(self, random_state=None, k_folds=10, n_jobs=1, verbose=0):
        super().__init__(random_state=random_state, n_repetitions=1, k_folds=k_folds, n_jobs=n_jobs, keep_features=False, verbose=verbose)


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
