import multiprocessing
import os
import threading
import pickle
import shutil
import uuid
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import polars as pl
from aeon.classification.base import BaseClassifier
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.interval_based import RSTSF
from aeon.regression.base import BaseRegressor
from aeon.transformations.collection import (
    ARCoefficientTransformer,
    PeriodogramTransformer,
)
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.transformations.collection.interval_based import QUANTTransformer, RandomIntervals
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from aeon.utils.numba.general import first_order_differences_3d
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from threadpoolctl import threadpool_limits

from tscglue import utils
from tscglue.utils import RidgeClassifierCVDecisionProba


class RareClassSafeLogisticCV(BaseEstimator, ClassifierMixin):
    """LogisticRegressionCV that won't crash on rare classes.

    LogisticRegressionCV runs an internal stratified CV to choose C. When a class
    has a single member in the data it is handed, one fold's training split omits
    that class and the multinomial coefficient paths become ragged, so the
    internal ``np.reshape`` raises "inhomogeneous shape". This wrapper drops the
    internal CV (fixed-C ``LogisticRegression``) only in that singleton case and
    is otherwise bit-identical to the original ``LogisticRegressionCV``.
    """

    def __init__(self, Cs=10, fixed_C=1.0, solver="lbfgs", max_iter=1000):
        self.Cs = Cs
        self.fixed_C = fixed_C
        self.solver = solver
        self.max_iter = max_iter

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

        min_count = int(np.unique(y, return_counts=True)[1].min())
        if min_count < 2:
            # A singleton class would make the internal CV crash; skip it.
            self.estimator_ = LogisticRegression(
                C=self.fixed_C,
                solver=self.solver,
                max_iter=self.max_iter,
                multi_class="multinomial",
            )
        else:
            # Identical to the original stacker: default cv=5, multinomial.
            self.estimator_ = LogisticRegressionCV(
                Cs=self.Cs,
                solver=self.solver,
                max_iter=self.max_iter,
                multi_class="multinomial",
            )
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class AutoSelectKBestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, classifier=None, k=None, k_min=6000, k_max=35000, midpoint=300, steepness=0.010
    ):
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


def get_model_v6(name, seed=None, n_jobs=1, model_dir=None, **kwargs):
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
            n_estimators=200,
            max_features=0.1,
            criterion="entropy",
            random_state=seed,
            n_jobs=n_jobs,
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
    elif name == "probability-logisticcv":
        scaler = DictMultiScaler(scalers={"probabilities": StandardScaler()})
        clf = RareClassSafeLogisticCV(Cs=np.logspace(-3, 3, 20))
        return scaler, clf
    elif name == "probability-tabicl":
        from tabicl import TabICLClassifier

        scaler = DictMultiScaler(scalers={"probabilities": NoScaler()})
        clf = TabICLClassifier(device="cpu", random_state=seed, n_jobs=1, kv_cache=False)
        return scaler, clf
    elif name == "probability-et":
        scaler = DictMultiScaler(scalers={"probabilities": NoScaler()})
        clf = ExtraTreesClassifier(
            n_estimators=1000,
            random_state=seed,
            n_jobs=n_jobs,
        )
        return scaler, clf
    elif name == "probability-rf":
        scaler = DictMultiScaler(scalers={"probabilities": NoScaler()})
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
        return scaler, clf
    elif name == "probability-nn":
        from sklearn.neural_network import MLPClassifier

        scaler = DictMultiScaler(scalers={"probabilities": StandardScaler()})
        clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=seed)
        return scaler, clf
    elif name == "probability-autogluon":
        import tempfile

        from autogluon.tabular.experimental import TabularClassifier

        ag_path = (
            str(Path(model_dir) / f"ag_{uuid.uuid4().hex[:8]}") if model_dir else tempfile.mkdtemp()
        )
        scaler = DictMultiScaler(scalers={"probabilities": NoScaler()})
        clf = TabularClassifier(
            path=ag_path,
            time_limit=kwargs.get("ag_time_limit", 60),
            presets=kwargs.get("ag_preset", "medium_quality"),
            verbosity=0,
        )
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
    elif name == "rstsf-random-etc":
        scaler = DictMultiScaler(scalers={"rstsf-random": NoScaler()})
        clf = ExtraTreesClassifier(
            n_estimators=200,
            criterion="entropy",
            class_weight="balanced",
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=seed,
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
    os.makedirs(directory, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    size = os.path.getsize(path)
    return path, size


def read_model(name, directory, repetition=None, fold=None):
    directory = str(directory)
    rep_suffix = f"_r_{repetition}" if repetition is not None else ""
    fold_suffix = f"_f_{fold}" if fold is not None else ""
    path = f"{directory}/{name}{rep_suffix}{fold_suffix}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


class RDSTFloat64(RandomDilatedShapeletTransform):
    """RDST wrapper that casts input to float64 (numba requires it)."""

    def _fit(self, X, y=None):
        return super()._fit(np.asarray(X, dtype=np.float64), y)

    def _transform(self, X, y=None):
        return super()._transform(np.asarray(X, dtype=np.float64), y)


def get_feature_transformer(feature_type: str, seed: int, n_jobs: int = 1, device: str = "cpu"):
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

            return MantisEmbedding(device=device)
        case "chronos2":
            from tscglue.models_tsfm import Chronos2Embedding

            return Chronos2Embedding(device=device)
        case "tsfresh":
            from aeon.transformations.collection.feature_based import TSFresh

            return TSFresh(default_fc_parameters="efficient", n_jobs=n_jobs)
        case "rstsf-random":
            from tscglue.interval_models import RSTSFRandomTransformer

            return RSTSFRandomTransformer(n_jobs=n_jobs, random_state=seed)
        case "drcif":
            from tscglue.drcif_features import DrCIFExtractor

            return DrCIFExtractor(random_state=seed, n_jobs=n_jobs)
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


def _fit_transformer_in_subprocess(
    feature_name, feature_seed, n_jobs, X_path, model_dir, feature_id
):
    X = np.load(X_path, allow_pickle=True)
    transformer = get_feature_transformer(feature_name, seed=feature_seed, n_jobs=n_jobs)
    transformer.fit(X)
    save_model(transformer, f"transformer_{feature_id}", model_dir)


def _transform_in_subprocess(
    feature_id, X_path, model_dir, output_dir, dtype=np.float64, verbose=0
):
    X = np.load(X_path, allow_pickle=True)
    transformer = read_model(f"transformer_{feature_id}", model_dir)
    t0 = perf_counter()
    Xt = transformer.transform(X)
    if verbose >= 3:
        print(f"[subprocess] transform {feature_id}: {perf_counter() - t0:.4f}s")
    save_array(Xt, f"Xt_{feature_id}", output_dir, dtype=dtype)


def _fit_transform_in_subprocess(
    feature_name,
    feature_seed,
    n_jobs,
    X_path,
    model_dir,
    output_dir,
    feature_id,
    dtype=np.float64,
    verbose=0,
    device="cpu",
):
    X = np.load(X_path, allow_pickle=True)
    transformer = get_feature_transformer(
        feature_name, seed=feature_seed, n_jobs=n_jobs, device=device
    )
    t0 = perf_counter()
    Xt = transformer.fit_transform(X)
    if verbose >= 3:
        print(f"[subprocess] fit_transform {feature_id}: {perf_counter() - t0:.4f}s")
    save_model(transformer, f"transformer_{feature_id}", model_dir)
    save_array(Xt, f"Xt_{feature_id}", output_dir, dtype=dtype)


def _transform_inline(feature_id, X_path, model_dir, output_dir, dtype=np.float64):
    X = np.load(X_path, allow_pickle=True)
    transformer = read_model(f"transformer_{feature_id}", model_dir)
    Xt = transformer.transform(X)
    save_array(Xt, f"Xt_{feature_id}", output_dir, dtype=dtype)


def _fit_transform_inline(
    feature_name,
    feature_seed,
    n_jobs,
    X_path,
    model_dir,
    output_dir,
    feature_id,
    dtype=np.float64,
    device="cpu",
):
    X = np.load(X_path, allow_pickle=True)
    transformer = get_feature_transformer(
        feature_name, seed=feature_seed, n_jobs=n_jobs, device=device
    )
    Xt = transformer.fit_transform(X)
    save_model(transformer, f"transformer_{feature_id}", model_dir)
    save_array(Xt, f"Xt_{feature_id}", output_dir, dtype=dtype)


@dataclass(frozen=True)
class FeatureSpec:
    feature_name: str
    feature_seed: int | None = None
    use_subprocess: bool = True

    def get_feature_id(self):
        return (
            f"{self.feature_name}_s_{self.feature_seed}"
            if self.feature_seed is not None
            else self.feature_name
        )


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_seed: int
    is_series: bool
    level: int
    features: tuple[FeatureSpec, ...]
    fold_seeds: tuple[int, ...]

    def get_model_id(self):
        return f"{self.model_name}_s_{self.model_seed}"

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


def _predict_one_model_v10(
    model_id, model_name, is_series, directory, feature_specs, model_dir, fold
):
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


def _train_one_model_v10(
    fold_number,
    model_id,
    model_name,
    is_series,
    train_idx,
    val_idx,
    model_seed,
    directory,
    feature_specs,
    model_dir,
    model_kwargs=None,
):
    """Training function - loads data via read_array, saves model to disk."""
    X = read_array("X", directory)
    y = read_array("y", directory)
    feature_dict = _load_feature_dict_v10(directory, feature_specs)

    scaler, clf = get_model_v6(model_name, seed=model_seed, model_dir=model_dir, **(model_kwargs or {}))
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
        "multirockethydra-bestk-p-ridgecv",
        "quant-etc",
        "rdst-p-ridgecv",
        "rstsf",
    ]
    SERIES_MODELS = ["rstsf"]
    STACKING_MODEL = "probability-ridgecv"
    NO_SUBPROCESS_FEATURES: set[str] = {"multirocket", "rdst"}
    NO_SUBPROCESS_FEATURES: set[str] = {"multirocket", "rdst"}

    def _get_feature_names(self, model_name: str) -> tuple[str, ...]:
        """Return required feature type names for a model."""
        if model_name in (
            "multirockethydra-bestk-p-ridgecv",
            "multirockethydra-p-ridgecv",
            "multirockethydra-ridgecv",
        ):
            return ("multirocket", "hydra")
        elif model_name == "quant-etc":
            return ("quant",)
        elif model_name in ("rdst-p-ridgecv", "rdst-ridgecv"):
            return ("rdst",)
        elif model_name == "rstsf":
            return ("raw",)
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
        use_subprocess = feature_name not in self.NO_SUBPROCESS_FEATURES
        if feature_name in ("quant", "raw", "mantis", "chronos2", "tsfresh"):
            return FeatureSpec(feature_name=feature_name, use_subprocess=use_subprocess)
        return FeatureSpec(
            feature_name=feature_name,
            feature_seed=int(group_rng.integers(0, 2**31 - 1)),
            use_subprocess=use_subprocess,
        )

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

    def __init__(
        self,
        random_state=None,
        k_folds=10,
        n_jobs=1,
        keep_features=False,
        verbose=0,
        model_names=None,
        n_repetitions=1,
        feature_dtype=None,
        stacking_models=None,
        selection=None,
        n_gpus=0,
        runs_dir=None,
        ag_preset=None,
        ag_time_limit=None,
        eval_metric="accuracy",
    ):
        super().__init__()
        self.k_folds = int(k_folds)
        self.random_state = random_state
        self.n_jobs = int(n_jobs)
        self.n_gpus = int(n_gpus)
        self.keep_features = bool(keep_features)
        self.verbose = int(verbose)
        self.n_repetitions = int(n_repetitions)
        self.feature_dtype = np.dtype(feature_dtype) if feature_dtype is not None else None
        self.stacking_models = (
            stacking_models if stacking_models is not None else [self.STACKING_MODEL]
        )
        self.selection = selection
        self.runs_dir = runs_dir
        self.ag_preset = ag_preset
        self.ag_time_limit = ag_time_limit
        self.eval_metric = eval_metric

        self.cv_splits = None
        self.feature_seed = np.random.default_rng(random_state)

        self._run_id = uuid.uuid4().hex[:16]
        self._base_dir = Path(
            ".", runs_dir if runs_dir is not None else "tscglue_runs", self._run_id
        )
        self._model_dir = self._base_dir / "models"
        self._tmpdir: Path | None = self._base_dir / "features_training"

        self.model_names = model_names
        self.series_models = self.SERIES_MODELS.copy()

        # Build model specs from flat list; derive unique features
        self.model_specs = self.build_model_specs(
            self.model_names if self.model_names is not None else self.DEFAULT_MODEL_NAMES
        )
        self.best_model = (
            self.stacking_models[0] if self.stacking_models else self.model_specs[0].get_model_id()
        )
        all_features: dict[str, FeatureSpec] = {}
        for spec in self.model_specs:
            for ft in spec.features:
                fid = ft.get_feature_id()
                if fid not in all_features:
                    all_features[fid] = ft
        self.features_list = list(all_features.values())

        self._oof_scores: list[dict] = []
        self._transform_times: list[dict] = []
        self._probability_columns: list[tuple[int, str, Any]] | None = None

        self._fallback_path: Path = self._model_dir / "fallback.pkl"

    # ----------------- utils -----------------

    @property
    def _device(self) -> str:
        return "cuda" if self.n_gpus != 0 else "cpu"

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

    def _probability_key(self, level: int, model_name: str, cls: Any) -> tuple[int, str, Any]:
        return int(level), model_name, self._label_to_python(cls)

    def _probability_sort_key(self, key: tuple[int, str, Any]) -> tuple[int, str, Any]:
        level, model_name, cls = key
        return level, model_name, self._label_to_python(cls)

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
        classes = sorted({self._label_to_python(p["class"]) for p in model_preds})
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

    def _compute_oof_score(self, y, model_name) -> float:
        prob_array, _level, classes = self._load_model_predictions(model_name)
        valid = ~np.isnan(prob_array).any(axis=1)
        if not np.any(valid):
            return 0.0
        y_true = y[np.where(valid)[0]]
        proba = prob_array[valid]

        if self.eval_metric == "accuracy":
            pred_idx = np.argmax(proba, axis=1)
            preds = np.asarray(classes)[pred_idx]
            return float(accuracy_score(np.asarray(y_true, dtype=str), np.asarray(preds, dtype=str)))
        elif self.eval_metric == "log_loss":
            from sklearn.metrics import log_loss
            return float(log_loss(y_true, proba, labels=classes))
        elif self.eval_metric == "roc_auc":
            from sklearn.metrics import roc_auc_score
            if len(classes) == 2:
                return float(roc_auc_score(y_true, proba[:, 1]))
            return float(roc_auc_score(y_true, proba, multi_class="ovr", labels=classes))
        elif self.eval_metric == "average_precision":
            from sklearn.metrics import average_precision_score
            from sklearn.preprocessing import label_binarize
            if len(classes) == 2:
                return float(average_precision_score(y_true, proba[:, 1]))
            y_bin = label_binarize(y_true, classes=classes)
            return float(average_precision_score(y_bin, proba, average="macro"))
        else:
            raise ValueError(f"Unknown eval_metric: {self.eval_metric!r}")

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
        """Fit transformers and compute features.

        When a GPU is available, GPU-bound features (use_subprocess=True) run in a
        background thread while CPU features run on the main thread, so both processors
        are used simultaneously. When no GPU is available all features run sequentially
        on the main thread, identical to the previous behaviour.
        """
        os.makedirs(self._model_dir, exist_ok=True)
        directory = str(self._tmpdir)
        X_path = str(self._tmpdir / "X.npy")

        _GPU_FEATURE_NAMES = {"mantis", "chronos2"}
        use_gpu = self._device != "cpu"
        gpu_features = [ft for ft in self.features_list if ft.feature_name != "raw" and ft.feature_name in _GPU_FEATURE_NAMES and use_gpu]
        cpu_features = [ft for ft in self.features_list if ft.feature_name != "raw" and (ft.feature_name not in _GPU_FEATURE_NAMES or not use_gpu)]
        gpu_error: list[BaseException] = []

        def _log_feature(ft, t0):
            Xt = read_array(f"Xt_{ft.get_feature_id()}", directory)
            size_mb = Xt.nbytes / (1024 * 1024)
            elapsed = perf_counter() - t0
            self.log(
                f"Fit+transformed {ft.get_feature_id()} features {Xt.shape} ({size_mb:.2f} MB) dtype={Xt.dtype} in {elapsed:.4f}s",
                level=1,
                start_time=fit_start_time,
            )
            self._transform_times.append(
                {
                    "model": ft.get_feature_id(),
                    "level": None,
                    "oof_accuracy": None,
                    "train_time": [elapsed],
                }
            )

        def _run_gpu_queue():
            try:
                for ft in gpu_features:
                    t0 = perf_counter()
                    _run_in_subprocess(
                        _fit_transform_in_subprocess,
                        (
                            ft.feature_name,
                            ft.feature_seed,
                            self.n_jobs,
                            X_path,
                            str(self._model_dir),
                            directory,
                            ft.get_feature_id(),
                            self.feature_dtype,
                            self.verbose,
                            self._device,
                        ),
                    )
                    _log_feature(ft, t0)
            except Exception as e:
                gpu_error.append(e)

        gpu_thread = threading.Thread(target=_run_gpu_queue, daemon=True)
        gpu_thread.start()

        for ft in cpu_features:
            t0 = perf_counter()
            if ft.use_subprocess:
                _run_in_subprocess(
                    _fit_transform_in_subprocess,
                    (
                        ft.feature_name,
                        ft.feature_seed,
                        self.n_jobs,
                        X_path,
                        str(self._model_dir),
                        directory,
                        ft.get_feature_id(),
                        self.feature_dtype,
                        self.verbose,
                        self._device,
                    ),
                )
            else:
                _fit_transform_inline(
                    ft.feature_name,
                    ft.feature_seed,
                    self.n_jobs,
                    X_path,
                    str(self._model_dir),
                    directory,
                    ft.get_feature_id(),
                    self.feature_dtype,
                    self._device,
                )
            _log_feature(ft, t0)

        gpu_thread.join()
        if gpu_error:
            raise RuntimeError("GPU feature extraction failed") from gpu_error[0]

    def compute_features(self, X: np.ndarray, directory: str, start_time=None) -> None:
        compute_start = perf_counter()
        X_path = f"{directory}/X.npy"

        _GPU_FEATURE_NAMES = {"mantis", "chronos2"}
        use_gpu = self._device != "cpu"
        gpu_features = [ft for ft in self.features_list if ft.feature_name != "raw" and ft.feature_name in _GPU_FEATURE_NAMES and use_gpu]
        cpu_features = [ft for ft in self.features_list if ft.feature_name != "raw" and (ft.feature_name not in _GPU_FEATURE_NAMES or not use_gpu)]
        gpu_error: list[BaseException] = []

        def _log_feature(ft, t0):
            Xt = read_array(f"Xt_{ft.get_feature_id()}", directory)
            size_mb = Xt.nbytes / (1024 * 1024)
            self.log(
                f"Computed {ft.get_feature_id()} features {Xt.shape} ({size_mb:.2f} MB) dtype={Xt.dtype} in {perf_counter() - t0:.4f}s",
                level=1,
                start_time=compute_start if start_time is None else start_time,
            )

        def _run_gpu_queue():
            try:
                for ft in gpu_features:
                    t0 = perf_counter()
                    _run_in_subprocess(
                        _transform_in_subprocess,
                        (
                            ft.get_feature_id(),
                            X_path,
                            str(self._model_dir),
                            directory,
                            self.feature_dtype,
                            self.verbose,
                        ),
                    )
                    _log_feature(ft, t0)
            except Exception as e:
                gpu_error.append(e)

        gpu_thread = threading.Thread(target=_run_gpu_queue, daemon=True)
        gpu_thread.start()

        for ft in cpu_features:
            t0 = perf_counter()
            if ft.use_subprocess:
                _run_in_subprocess(
                    _transform_in_subprocess,
                    (
                        ft.get_feature_id(),
                        X_path,
                        str(self._model_dir),
                        directory,
                        self.feature_dtype,
                        self.verbose,
                    ),
                )
            else:
                _transform_inline(
                    ft.get_feature_id(),
                    X_path,
                    str(self._model_dir),
                    directory,
                    self.feature_dtype,
                )
            _log_feature(ft, t0)

        gpu_thread.join()
        if gpu_error:
            raise RuntimeError("GPU feature extraction failed") from gpu_error[0]

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
        self.log(
            f"Starting fit, run_dir={self._base_dir}, n_jobs={self.n_jobs}",
            level=1,
            start_time=fit_start,
        )
        _cpu_max = os.cpu_count() or 1
        _cpu_used = _cpu_max if self.n_jobs == -1 else self.n_jobs
        self.log(
            f"CPUs set/available/used/ {_cpu_used}/{_cpu_max}/{_cpu_used}",
            level=1,
            start_time=fit_start,
        )
        try:
            import torch

            _gpu_torch = torch.cuda.device_count()
        except Exception:
            _gpu_torch = 0
        try:
            import subprocess

            _gpu_smi = len(
                subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
                .splitlines()
            )
        except Exception:
            _gpu_smi = 0
        _gpu_used = 1 if self.n_gpus != 0 else 0
        self.log(
            f"GPUs set/available[torch]/available[smi]/used/ {_gpu_used}/{_gpu_torch}/{_gpu_smi}/{_gpu_used}",
            level=1,
            start_time=fit_start,
        )
        _direction = "minimize" if self.eval_metric == "log_loss" else "maximize"
        self.log(
            f"Eval metric: {self.eval_metric} ({_direction})",
            level=1,
            start_time=fit_start,
        )

        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._tmpdir, exist_ok=True)

        t0 = perf_counter()
        save_array(X, "X", str(self._tmpdir), dtype=self.feature_dtype)
        save_array(y, "y", str(self._tmpdir))
        self.log(
            f"Saved X and y to disk in {perf_counter() - t0:.2f}s (dtype={self.feature_dtype})",
            level=2,
            start_time=fit_start,
        )

        _, counts = np.unique(y, return_counts=True)
        if np.any(counts < 2):
            self.log(
                "Some classes have fewer than 2 instances, fold training not possible",
                level=1,
                start_time=fit_start,
            )
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
                expected_folds = {
                    spec.get_model_id(): self.k_folds * spec.n_repetitions
                    for spec in self.model_specs
                }
                tasks = []
                for spec in self.model_specs:
                    fold_rng = np.random.default_rng(spec.model_seed)
                    fold_counter = 0
                    for fold_seed in spec.fold_seeds:
                        rep_splits = generate_folds(
                            X, y, n_splits=self.k_folds, n_repetitions=1, random_state=fold_seed
                        )
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
                self.log(
                    f"Starting training with {n_workers} workers for {len(tasks)} models",
                    level=2,
                    start_time=fit_start,
                )

                futures = {executor.submit(_train_one_model_v10, *t): t for t in tasks}
                model_groups = defaultdict(list)
                model_train_times: dict[str, list[float]] = defaultdict(list)

                for future in as_completed(futures):
                    task = futures[future]
                    fold_number = task[0]
                    model_id_task = task[1]
                    try:
                        (
                            train_idx,
                            val_idx,
                            proba,
                            classes_,
                            model_size,
                            train_dur,
                            model_id_result,
                            fold_number,
                        ) = future.result()
                    except Exception as e:
                        raise RuntimeError(
                            f"Worker failed during training {model_id_task} fold {fold_number}: {e}"
                        ) from e

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

                    model_groups[model_id_result].append(fold_number)
                    model_train_times[model_id_result].append(train_dur)
                    if len(model_groups[model_id_result]) == expected_folds[model_id_result]:
                        self.log(
                            f"Completed training for model {model_id_result}",
                            level=2,
                            start_time=fit_start,
                        )
                        del model_groups[model_id_result]

                        predictions = self._save_model_predictions(
                            predictions, model_id_result, n_samples=X.shape[0], level=0
                        )
                        oof_score = self._compute_oof_score(y, model_id_result)
                        self._oof_scores.append(
                            {
                                "model": model_id_result,
                                "level": 0,
                                "eval_metric": self.eval_metric,
                                "oof_score": oof_score,
                                "train_time": model_train_times.pop(model_id_result),
                            }
                        )
                        self.log(
                            f"OOF {self.eval_metric} (base) {model_id_result}: {oof_score}",
                            level=1,
                            start_time=fit_start,
                        )

                # -------- stacking --------
                prob_array = self._build_probability_array(n_samples=X.shape[0])
                if not self.stacking_models:
                    return
                self.log("Starting stacking model training", level=2, start_time=fit_start)
                if prob_array is None or np.isnan(prob_array).any():
                    self.log(
                        "NaN values detected in probability array, skipping stacking",
                        level=2,
                        start_time=fit_start,
                    )
                    self._fit_fallback(X, y, fit_start)
                    return

                save_array(prob_array, "Xt_probabilities", str(self._tmpdir))

                stacker_splits = generate_folds(
                    X, y, n_splits=self.k_folds, n_repetitions=1, random_state=stacker_fold_seed
                )
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
                self.log(
                    f"Starting stacking training with {n_workers} workers for {len(stack_tasks)} models",
                    level=2,
                    start_time=fit_start,
                )

                ag_kwargs = {"ag_preset": self.ag_preset, "ag_time_limit": self.ag_time_limit}
                futures = {executor.submit(_train_one_model_v10, *t, model_kwargs=ag_kwargs): t for t in stack_tasks}
                model_groups = defaultdict(list)
                model_train_times: dict[str, list[float]] = defaultdict(list)

                for future in as_completed(futures):
                    task = futures[future]
                    fold_number = task[0]
                    model_id_task = task[1]
                    try:
                        (
                            train_idx,
                            val_idx,
                            proba,
                            classes_,
                            model_size,
                            train_dur,
                            model_id_result,
                            fold_number,
                        ) = future.result()
                    except Exception as e:
                        raise RuntimeError(
                            f"Worker failed during stacking training {model_id_task} fold {fold_number}: {e}"
                        ) from e

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

                    model_groups[model_id_result].append(fold_number)
                    model_train_times[model_id_result].append(train_dur)
                    if len(model_groups[model_id_result]) == self.k_folds:
                        self.log(
                            f"Completed training for model {model_id_result}",
                            level=2,
                            start_time=fit_start,
                        )
                        del model_groups[model_id_result]

                        predictions = self._save_model_predictions(
                            predictions, model_id_result, n_samples=X.shape[0], level=1
                        )
                        oof_score = self._compute_oof_score(y, model_id_result)
                        self._oof_scores.append(
                            {
                                "model": model_id_result,
                                "level": 1,
                                "eval_metric": self.eval_metric,
                                "oof_score": oof_score,
                                "train_time": model_train_times.pop(model_id_result),
                            }
                        )
                        self.log(
                            f"OOF {self.eval_metric} (stack) {model_id_result}: {oof_score}",
                            level=1,
                            start_time=fit_start,
                        )

                self.log("Fit complete", level=1, start_time=fit_start)
                self._select_best_model()

        finally:
            if not self.keep_features and self._tmpdir and self._tmpdir.exists():
                cleanup_start = perf_counter()
                shutil.rmtree(self._tmpdir)
                self.log(
                    f"Cleaned up tmpdir in {perf_counter() - cleanup_start:.2f}s",
                    level=2,
                    start_time=fit_start,
                )
                self._tmpdir = None
            if self.keep_features and self._tmpdir:
                self.features_training_dir_ = str(self._tmpdir)
            self.log("Executor shutdown complete", level=2, start_time=fit_start)

    def _select_best_model(self):
        if self.selection is None:
            return
        if self.selection == "best":
            candidates = self._oof_scores
        elif self.selection == "best-stacking":
            candidates = [s for s in self._oof_scores if s["level"] == 1]
        elif self.selection == "best-base":
            candidates = [s for s in self._oof_scores if s["level"] == 0]
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection!r}")
        if not candidates:
            return
        higher_is_better = self.eval_metric != "log_loss"
        self.best_model = (max if higher_is_better else min)(
            candidates, key=lambda s: s["oof_score"]
        )["model"]
        self.log(f"Selected best model ({self.selection}): {self.best_model}", level=1)

    # ----------------- inspection helpers -----------------

    def _get_training_dir(self) -> str:
        d = getattr(self, "features_training_dir_", None) or (
            str(self._tmpdir) if self._tmpdir else None
        )
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

    def summary(self, return_transforms: bool = False) -> list[dict]:
        if return_transforms:
            return self._transform_times + self._oof_scores
        return self._oof_scores

    def optimize_for_inference(self, drop_unused_features: bool = True) -> None:
        """Print how many features are never selected across any fold, per feature transform."""
        if not drop_unused_features:
            return

        def _print_line(name, n_used, n_total):
            n_unused = n_total - n_used
            pct = 100 * n_unused / n_total if n_total else 0.0
            print(f"  {name + ':':<14}{n_used}/{n_total} used, {n_unused} unused ({pct:.1f}%)")

        def _group_counts(scaler, clf):
            """Feature count per scaler group; NoScaler groups inferred from clf input dim."""
            known, unknown = {}, []
            for key, s in scaler.scalers_.items():
                if hasattr(s, "n_features_in_"):
                    known[key] = int(s.n_features_in_)
                elif hasattr(s, "mu"):
                    known[key] = len(s.mu)
                else:
                    unknown.append(key)
            total = int(getattr(clf, "n_features_in_", sum(known.values())))
            if len(unknown) == 1:
                known[unknown[0]] = total - sum(known.values())
            else:
                for key in unknown:
                    known[key] = 0
            return known

        for spec in self.model_specs:
            model_id = spec.get_model_id()
            print(f"{model_id}:")
            if spec.model_name == "multirockethydra-bestk-p-ridgecv":
                n_folds = self.k_folds * spec.n_repetitions
                hydra_union: set[int] = set()
                multirocket_union: set[int] = set()
                n_hydra = n_multirocket = None
                for fold in range(n_folds):
                    scaler, clf = read_model(model_id, str(self._model_dir), None, fold)
                    if n_hydra is None:
                        n_hydra = len(scaler.scalers_["hydra"].mu)
                        n_multirocket = int(scaler.scalers_["multirocket"].n_features_in_)
                    pipe = clf.classifier_
                    var_ix = pipe["var"].get_support(indices=True)
                    sel_ix = pipe["select"].get_support(indices=True)
                    final = var_ix[sel_ix]
                    hydra_union.update(final[final < n_hydra].tolist())
                    multirocket_union.update((final[final >= n_hydra] - n_hydra).tolist())
                _print_line("hydra", len(hydra_union), n_hydra)
                _print_line("multirocket", len(multirocket_union), n_multirocket)
                print("  (dropping unused features not yet implemented — kernel-to-feature mapping required)")
            else:
                scaler, clf = read_model(model_id, str(self._model_dir), None, 0)
                for key, n in _group_counts(scaler, clf).items():
                    _print_line(key, n, n)

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
                self.log(
                    "Computed and saved features for prediction", level=1, start_time=predict_start
                )

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
                        raise RuntimeError(
                            f"Worker failed during prediction {model_id_task}: {e}"
                        ) from e

                    self.log(
                        f"Predicted {model_id_res} in {predict_dur:.4f}s",
                        level=2,
                        start_time=predict_start,
                    )
                    predictions.extend(
                        self.add_probabilities(proba, classes_, model_id_res, level=0)
                    )

                self.log(
                    "Completed all first-level model predictions", level=1, start_time=predict_start
                )

                # ---- build stacking matrix ----
                if features_infer.exists():
                    shutil.rmtree(features_infer)
                os.makedirs(features_stack, exist_ok=True)
                self._tmpdir = features_stack

                if self._probability_columns is None:
                    raise RuntimeError(
                        "Probability column metadata missing. Fit the model before predicting."
                    )
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
                        raise RuntimeError(
                            f"Worker failed during stacking prediction {model_id_task}: {e}"
                        ) from e

                    self.log(
                        f"Predicted {model_id_res} in {predict_dur:.4f}s",
                        level=2,
                        start_time=predict_start,
                    )
                    predictions.extend(
                        self.add_probabilities(proba, classes_, model_id_res, level=1)
                    )

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
        return {
            name: self.classes_[np.argmax(proba, axis=1)] for name, proba in proba_per_model.items()
        }


class LokyStackerV10RSTSFRandom(LokyStackerV10Base):
    """Splits RSTSF into a feature extraction transformer (RandomIntervals on 4 series
    representations) and a separately trained ExtraTreesClassifier. SERIES_MODELS is
    empty so all models go through the feature caching pipeline.
    """

    DEFAULT_MODEL_NAMES = [
        "multirockethydra-bestk-p-ridgecv",
        "quant-etc",
        "rdst-p-ridgecv",
        "rstsf-random-etc",
        "fm-p-ridgecv",
    ]
    SERIES_MODELS = []
    NO_SUBPROCESS_FEATURES: set[str] = {"multirocket", "rdst", "rstsf-random"}


class LokyStackerV10RSTSFRandomMultiStack(LokyStackerV10RSTSFRandom):
    """LokyStackerV10RSTSFRandom with multiple stacking models for best-stacking selection experiments."""

    STACKING_MODEL = "probability-ridgecv"

    def __init__(
        self, random_state=None, k_folds=10, n_jobs=1, verbose=0, selection="best-stacking"
    ):
        super().__init__(
            random_state=random_state,
            k_folds=k_folds,
            n_jobs=n_jobs,
            verbose=verbose,
            stacking_models=[
                "probability-ridgecv",
                "probability-et",
                "probability-nn",
                "probability-rf",
            ],
            selection=selection,
        )


def _robust_r2(y, pred):
    """Outlier-robust R²: ordinary R² on predictions clipped to the target range.

    Standard R² is dominated by its single largest squared residual, so one
    off-scale prediction (a high-leverage ridge extrapolation, e.g. -4 on a
    target in [0, 0.18]) can drive it to large negative values even when the
    model is good on the other 99% of samples. Clipping predictions to
    [min(y), max(y)] before scoring neutralises such nonsensical values — and is
    a no-op for well-behaved models (ETR, clipped variants), so it only changes
    the pathological cases. It also matches how ``ClippedRegressor`` actually
    serves predictions, and stays on the same scale as ``r2_score``.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(r2_score(y, np.clip(pred, np.nanmin(y), np.nanmax(y))))


def generate_folds(X, y, n_splits=5, n_repetitions=5, random_state=0, stratify=True):
    all_folds = []
    for i in range(n_repetitions):
        folds = utils.get_folds(
            X, y, n_splits=n_splits, random_state=random_state + i, stratify=stratify
        )
        all_folds.extend(folds)
    return all_folds


_VALID_EVAL_METRICS = {"accuracy", "log_loss", "roc_auc"}


class TSCGlueClassifier(LokyStackerV10RSTSFRandom):
    def __init__(
        self,
        random_state=None,
        k_folds=10,
        n_jobs=1,
        verbose=0,
        n_repetitions=1,
        n_gpus=0,
        runs_dir=None,
        eval_metric="accuracy",
        time_limit=None,
    ):
        assert n_gpus in (0, 1, -1), f"n_gpus must be 0, 1, or -1; got {n_gpus}"
        assert eval_metric in _VALID_EVAL_METRICS, f"eval_metric must be one of {_VALID_EVAL_METRICS}; got {eval_metric!r}"
        assert time_limit is None, "time_limit is currently not supported"
        # Hardcoded best stacker per metric (from the critical-difference study):
        # ridge wins on accuracy; ExtraTrees wins log-loss and AUC.
        stacking = {
            "accuracy": ["probability-ridgecv"],
            "log_loss": ["probability-et"],
            "roc_auc": ["probability-et"],
        }[eval_metric]
        super().__init__(
            random_state=random_state,
            n_repetitions=n_repetitions,
            k_folds=k_folds,
            n_jobs=n_jobs,
            keep_features=False,
            verbose=verbose,
            n_gpus=n_gpus,
            runs_dir=runs_dir,
            stacking_models=stacking,
            eval_metric=eval_metric,
        )
        self.time_limit = time_limit


class TSCGlueLogisticClassifier(LokyStackerV10RSTSFRandom):
    STACKING_MODEL = "probability-logisticcv"

    def __init__(
        self,
        random_state=None,
        k_folds=10,
        n_jobs=1,
        verbose=0,
        n_repetitions=1,
        n_gpus=0,
        runs_dir=None,
        eval_metric="accuracy",
    ):
        assert n_gpus in (0, 1, -1), f"n_gpus must be 0, 1, or -1; got {n_gpus}"
        assert eval_metric in _VALID_EVAL_METRICS, f"eval_metric must be one of {_VALID_EVAL_METRICS}; got {eval_metric!r}"
        super().__init__(
            random_state=random_state,
            n_repetitions=n_repetitions,
            k_folds=k_folds,
            n_jobs=n_jobs,
            keep_features=False,
            verbose=verbose,
            n_gpus=n_gpus,
            runs_dir=runs_dir,
            stacking_models=["probability-logisticcv"],
            eval_metric=eval_metric,
        )


class TSCAGGlueClassifier(LokyStackerV10RSTSFRandom):
    def __init__(
        self,
        random_state=None,
        k_folds=10,
        n_jobs=1,
        verbose=0,
        n_repetitions=1,
        n_gpus=0,
        runs_dir=None,
        ag_preset="medium_quality",
        ag_time_limit=60,
    ):
        assert n_gpus in (0, 1, -1), f"n_gpus must be 0, 1, or -1; got {n_gpus}"
        super().__init__(
            random_state=random_state,
            n_repetitions=n_repetitions,
            k_folds=k_folds,
            n_jobs=n_jobs,
            keep_features=False,
            verbose=verbose,
            n_gpus=n_gpus,
            runs_dir=runs_dir,
            stacking_models=["probability-autogluon"],
            ag_preset=ag_preset,
            ag_time_limit=ag_time_limit,
        )


class AutoSelectKBestRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y):
        reg = (
            RidgeCV(alphas=np.logspace(5, 6, 13))
            if self.regressor is None
            else clone(self.regressor)
        )
        self.regressor_ = Pipeline(
            [
                ("var", VarianceThreshold()),
                ("reg", reg),
            ]
        )
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.regressor_.fit(X, y)
        return self

    def predict(self, X):
        return self.regressor_.predict(X)


class ClippedRegressor(BaseEstimator, RegressorMixin):
    """Wrap a regressor and clip predictions to the training target range.

    Lets a linear model (e.g. RidgeCV) keep its signal on ROCKET-style features
    while preventing the rare p >> n fold from extrapolating off-scale, the way
    tree models are bounded implicitly.
    """

    def __init__(self, regressor=None):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor_ = clone(self.regressor)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.regressor_.fit(X, y)
        self.y_min_ = float(np.min(y))
        self.y_max_ = float(np.max(y))
        return self

    def predict(self, X):
        return np.clip(self.regressor_.predict(X), self.y_min_, self.y_max_)


def get_model_reg(name, seed=None, n_jobs=1):
    if name == "multirockethydra-etr":
        scaler = DictMultiScaler(scalers={"hydra": SparseScaler(), "multirocket": StandardScaler()})
        # ExtraTrees instead of RidgeCV: trees predict a bounded average of seen
        # targets, so they can't extrapolate off-scale like ridge does at p >> n.
        return scaler, ExtraTreesRegressor(
            n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=n_jobs
        )
    elif name == "multirockethydra-ridgecv":
        scaler = DictMultiScaler(scalers={"hydra": SparseScaler(), "multirocket": StandardScaler()})
        return scaler, RidgeCV(alphas=np.logspace(-4, 14, 65))
    elif name == "multirockethydra-clipped-ridgecv":
        scaler = DictMultiScaler(scalers={"hydra": SparseScaler(), "multirocket": StandardScaler()})
        # RidgeCV keeps ROCKET's linear signal; clipping bounds predictions to the
        # training target range so a rare under-regularized fold can't blow up.
        return scaler, ClippedRegressor(regressor=RidgeCV(alphas=np.logspace(-4, 14, 65)))
    elif name == "quant-etr":
        scaler = DictMultiScaler(scalers={"quant": NoScaler()})
        return scaler, ExtraTreesRegressor(
            n_estimators=200, max_features=0.1, random_state=seed, n_jobs=n_jobs
        )
    elif name == "quant-ridgecv":
        scaler = DictMultiScaler(scalers={"quant": NoScaler()})
        return scaler, RidgeCV(alphas=np.logspace(-4, 14, 65))
    elif name == "quant-clipped-ridgecv":
        scaler = DictMultiScaler(scalers={"quant": NoScaler()})
        return scaler, ClippedRegressor(regressor=RidgeCV(alphas=np.logspace(-4, 14, 65)))
    elif name == "rdst-etr":
        scaler = DictMultiScaler(scalers={"rdst": StandardScaler()})
        return scaler, ExtraTreesRegressor(
            n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=n_jobs
        )
    elif name == "rdst-ridgecv":
        scaler = DictMultiScaler(scalers={"rdst": StandardScaler()})
        return scaler, RidgeCV(alphas=np.logspace(-4, 14, 65))
    elif name == "rdst-clipped-ridgecv":
        scaler = DictMultiScaler(scalers={"rdst": StandardScaler()})
        return scaler, ClippedRegressor(regressor=RidgeCV(alphas=np.logspace(-4, 14, 65)))
    elif name == "rstsf-random-etr":
        scaler = DictMultiScaler(scalers={"rstsf-random": NoScaler()})
        return scaler, ExtraTreesRegressor(
            n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=n_jobs
        )
    elif name == "rstsf-random-ridgecv":
        scaler = DictMultiScaler(scalers={"rstsf-random": NoScaler()})
        return scaler, RidgeCV(alphas=np.logspace(-4, 14, 65))
    elif name == "rstsf-random-clipped-ridgecv":
        scaler = DictMultiScaler(scalers={"rstsf-random": NoScaler()})
        return scaler, ClippedRegressor(regressor=RidgeCV(alphas=np.logspace(-4, 14, 65)))
    elif name == "fm-etr":
        scaler = DictMultiScaler(scalers={"mantis": StandardScaler(), "chronos2": StandardScaler()})
        return scaler, ExtraTreesRegressor(
            n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=n_jobs
        )
    elif name == "fm-ridgecv":
        scaler = DictMultiScaler(scalers={"mantis": StandardScaler(), "chronos2": StandardScaler()})
        return scaler, RidgeCV(alphas=np.logspace(-4, 14, 65))
    elif name == "fm-clipped-ridgecv":
        scaler = DictMultiScaler(scalers={"mantis": StandardScaler(), "chronos2": StandardScaler()})
        return scaler, ClippedRegressor(regressor=RidgeCV(alphas=np.logspace(-4, 14, 65)))
    elif name == "tsfresh-rotf":
        # FreshPRINCE: efficient TSFresh features + Rotation Forest regressor.
        from aeon.regression.sklearn import RotationForestRegressor

        scaler = DictMultiScaler(scalers={"tsfresh": NoScaler()})
        return scaler, RotationForestRegressor(
            n_estimators=200, n_jobs=n_jobs, random_state=seed
        )
    elif name == "drcif-etr":
        # DrCIF-like: fixed DrCIF interval features + a random-subspace ExtraTrees
        # whose per-split sqrt subsampling re-creates DrCIF's per-tree randomisation.
        scaler = DictMultiScaler(scalers={"drcif": NoScaler()})
        return scaler, ExtraTreesRegressor(
            n_estimators=200, max_features="sqrt", random_state=seed, n_jobs=n_jobs
        )
    elif name == "prediction-etr":
        scaler = DictMultiScaler(scalers={"predictions": StandardScaler()})
        return scaler, ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=n_jobs)
    else:
        raise ValueError(f"Unknown regressor model: {name}")


def _train_one_model_reg(
    fold_number,
    model_id,
    model_name,
    train_idx,
    val_idx,
    model_seed,
    directory,
    feature_specs,
    model_dir,
):
    X = read_array("X", directory)
    y = read_array("y", directory)
    feature_dict = _load_feature_dict_v10(directory, feature_specs)
    scaler, reg = get_model_reg(model_name, seed=model_seed)
    start_train = perf_counter()
    reg.fit(scaler.fit_transform(feature_dict, idx=train_idx), y[train_idx])
    preds = reg.predict(scaler.transform(feature_dict, idx=val_idx))
    _, model_size = save_model((scaler, reg), model_id, model_dir, None, fold_number)
    train_dur = perf_counter() - start_train
    return (train_idx, val_idx, preds, model_size, train_dur, model_id, fold_number)


def _predict_one_model_reg(model_id, directory, feature_specs, model_dir, fold):
    feature_dict = _load_feature_dict_v10(directory, feature_specs)
    scaler, reg = read_model(model_id, model_dir, None, fold)
    start = perf_counter()
    preds = reg.predict(scaler.transform(feature_dict))
    return (preds, perf_counter() - start, model_id)


class TSCGlueRegressor(BaseRegressor):
    _tags = {"capability:multivariate": True}
    DEFAULT_MODEL_NAMES = [
        "multirockethydra-etr",
        "multirockethydra-ridgecv",
        "multirockethydra-clipped-ridgecv",
        "quant-etr",
        "quant-ridgecv",
        "quant-clipped-ridgecv",
        "rdst-etr",
        "rdst-ridgecv",
        "rdst-clipped-ridgecv",
        "rstsf-random-etr",
        "rstsf-random-ridgecv",
        "rstsf-random-clipped-ridgecv",
        "fm-etr",
        "fm-ridgecv",
        "fm-clipped-ridgecv",
        "tsfresh-rotf",
        "drcif-etr",
    ]
    STACKING_MODEL = "prediction-etr"
    NO_SUBPROCESS_FEATURES: set[str] = {"multirocket", "rdst", "rstsf-random"}

    def _get_feature_names(self, model_name: str) -> tuple[str, ...]:
        if model_name in (
            "multirockethydra-etr",
            "multirockethydra-ridgecv",
            "multirockethydra-clipped-ridgecv",
        ):
            return ("multirocket", "hydra")
        elif model_name in ("quant-etr", "quant-ridgecv", "quant-clipped-ridgecv"):
            return ("quant",)
        elif model_name in ("rdst-etr", "rdst-ridgecv", "rdst-clipped-ridgecv"):
            return ("rdst",)
        elif model_name in (
            "rstsf-random-etr",
            "rstsf-random-ridgecv",
            "rstsf-random-clipped-ridgecv",
        ):
            return ("rstsf-random",)
        elif model_name in ("fm-etr", "fm-ridgecv", "fm-clipped-ridgecv"):
            return ("mantis", "chronos2")
        elif model_name == "tsfresh-rotf":
            return ("tsfresh",)
        elif model_name == "drcif-etr":
            return ("drcif",)
        else:
            raise ValueError(f"Unknown model {model_name}")

    def _make_feature_spec(self, feature_name: str, group_rng: np.random.Generator) -> FeatureSpec:
        use_subprocess = feature_name not in self.NO_SUBPROCESS_FEATURES
        if feature_name in ("quant", "mantis", "chronos2", "tsfresh"):
            return FeatureSpec(feature_name=feature_name, use_subprocess=use_subprocess)
        return FeatureSpec(
            feature_name=feature_name,
            feature_seed=int(group_rng.integers(0, 2**31 - 1)),
            use_subprocess=use_subprocess,
        )

    def _build_model_specs(self, model_names: list[str]) -> list[ModelSpec]:
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
            group_rng = np.random.default_rng(self._get_seed())
            group_features: dict[str, FeatureSpec] = {}
            for model_name in group:
                for ft_name in self._get_feature_names(model_name):
                    if ft_name not in group_features:
                        group_features[ft_name] = self._make_feature_spec(ft_name, group_rng)
            for model_name in group:
                features = tuple(
                    group_features[ft_name] for ft_name in self._get_feature_names(model_name)
                )
                model_seed = self._get_seed()
                fold_seed_rng = np.random.default_rng(model_seed)
                fold_seeds = tuple(
                    int(fold_seed_rng.integers(0, 2**31 - 1)) for _ in range(self.n_repetitions)
                )
                all_models.append(
                    ModelSpec(
                        model_name=model_name,
                        model_seed=model_seed,
                        is_series=False,
                        level=0,
                        features=features,
                        fold_seeds=fold_seeds,
                    )
                )
        return all_models

    def __init__(
        self,
        random_state=None,
        k_folds=10,
        n_jobs=1,
        verbose=0,
        n_repetitions=1,
        runs_dir=None,
        time_limit=None,
        drop_nonpositive_r2=True,
    ):
        assert time_limit is None, "time_limit is currently not supported"
        super().__init__()
        self.time_limit = time_limit
        self.random_state = random_state
        self.k_folds = int(k_folds)
        self.n_jobs = int(n_jobs)
        self.verbose = int(verbose)
        self.n_repetitions = int(n_repetitions)
        self.runs_dir = runs_dir
        # When True, base models whose OOF R² <= 0 (no better than predicting the
        # mean) are excluded from the stacking matrix so off-scale / no-skill
        # columns can't poison the stacker.
        self.drop_nonpositive_r2 = bool(drop_nonpositive_r2)

        self._rng = np.random.default_rng(random_state)
        self._run_id = uuid.uuid4().hex[:16]
        self._base_dir = Path(
            ".", runs_dir if runs_dir is not None else "tscglue_runs", self._run_id
        )
        self._model_dir = self._base_dir / "models"
        self._tmpdir: Path = self._base_dir / "features_training"
        self._feature_dtype: np.dtype | None = None

        self.stacking_models = [self.STACKING_MODEL]
        self.model_specs = self._build_model_specs(self.DEFAULT_MODEL_NAMES)
        all_features: dict[str, FeatureSpec] = {}
        for spec in self.model_specs:
            for ft in spec.features:
                fid = ft.get_feature_id()
                if fid not in all_features:
                    all_features[fid] = ft
        self.features_list = list(all_features.values())
        self._stacking_model_order: list[str] = []
        self._oof_scores: list[dict] = []
        self._transform_times: list[dict] = []

    def _get_seed(self) -> int:
        return int(self._rng.integers(0, 2**31 - 1, dtype=np.int32))

    def log(self, message: str, level: int, start_time=None, current_time=None):
        if self.verbose >= level:
            if start_time is not None:
                if current_time is None:
                    current_time = perf_counter()
                print(f"[{current_time - start_time:.2f}s] {message}")
            else:
                print(message)

    def summary(self, return_transforms: bool = False) -> list[dict]:
        if return_transforms:
            return self._transform_times + self._oof_scores
        return self._oof_scores

    def cleanup(self):
        if self._base_dir.exists():
            shutil.rmtree(self._base_dir)

    def _fit_transform_features(self, X: np.ndarray, fit_start_time=None) -> None:
        os.makedirs(self._model_dir, exist_ok=True)
        directory = str(self._tmpdir)
        X_path = str(self._tmpdir / "X.npy")
        for ft in self.features_list:
            if ft.feature_name == "raw":
                continue
            t0 = perf_counter()
            if ft.use_subprocess:
                _run_in_subprocess(
                    _fit_transform_in_subprocess,
                    (
                        ft.feature_name,
                        ft.feature_seed,
                        self.n_jobs,
                        X_path,
                        str(self._model_dir),
                        directory,
                        ft.get_feature_id(),
                        self._feature_dtype,
                        self.verbose,
                    ),
                )
            else:
                _fit_transform_inline(
                    ft.feature_name,
                    ft.feature_seed,
                    self.n_jobs,
                    X_path,
                    str(self._model_dir),
                    directory,
                    ft.get_feature_id(),
                    self._feature_dtype,
                )
            Xt = read_array(f"Xt_{ft.get_feature_id()}", directory)
            elapsed = perf_counter() - t0
            self.log(
                f"Fit+transformed {ft.get_feature_id()} features {Xt.shape} ({Xt.nbytes / (1024 * 1024):.2f} MB) dtype={Xt.dtype} in {elapsed:.4f}s",
                level=1,
                start_time=fit_start_time,
            )
            self._transform_times.append(
                {
                    "model": ft.get_feature_id(),
                    "level": None,
                    "oof_rmse": None,
                    "train_time": [elapsed],
                }
            )

    def _compute_features(self, X: np.ndarray, directory: str, start_time=None) -> None:
        X_path = f"{directory}/X.npy"
        for ft in self.features_list:
            if ft.feature_name == "raw":
                continue
            t0 = perf_counter()
            if ft.use_subprocess:
                _run_in_subprocess(
                    _transform_in_subprocess,
                    (
                        ft.get_feature_id(),
                        X_path,
                        str(self._model_dir),
                        directory,
                        self._feature_dtype,
                        self.verbose,
                    ),
                )
            else:
                _transform_inline(
                    ft.get_feature_id(),
                    X_path,
                    str(self._model_dir),
                    directory,
                    self._feature_dtype,
                )
            Xt = read_array(f"Xt_{ft.get_feature_id()}", directory)
            self.log(
                f"Computed {ft.get_feature_id()} features {Xt.shape} in {perf_counter() - t0:.4f}s",
                level=1,
                start_time=start_time,
            )

    def _fit(self, X, y):
        fit_start = perf_counter()
        self._feature_dtype = np.asarray(X).dtype

        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._tmpdir, exist_ok=True)
        save_array(X, "X", str(self._tmpdir), dtype=self._feature_dtype)
        save_array(y, "y", str(self._tmpdir))

        self._fit_transform_features(X, fit_start_time=fit_start)

        n_samples = X.shape[0]
        # One OOF vector per repetition (each repetition's k folds cover every
        # sample exactly once); combined across repetitions with the median so a
        # single extrapolating fold can't dominate the OOF estimate.
        oof_pred_mats = {
            spec.get_model_id(): np.full((spec.n_repetitions, n_samples), np.nan)
            for spec in self.model_specs
        }
        oof_preds: dict[str, np.ndarray] = {}
        expected_folds = {
            spec.get_model_id(): self.k_folds * spec.n_repetitions for spec in self.model_specs
        }
        base_oof_r2: dict[str, float] = {}

        tasks = []
        for spec in self.model_specs:
            fold_rng = np.random.default_rng(spec.model_seed)
            fold_counter = 0
            for fold_seed in spec.fold_seeds:
                for train_idx, val_idx in generate_folds(
                    X, y, n_splits=self.k_folds, n_repetitions=1, random_state=fold_seed,
                    stratify=False,
                ):
                    fold_model_seed = int(fold_rng.integers(0, 2**31 - 1))
                    tasks.append(
                        (
                            fold_counter,
                            spec.get_model_id(),
                            spec.model_name,
                            train_idx,
                            val_idx,
                            fold_model_seed,
                            str(self._tmpdir),
                            list(spec.features),
                            str(self._model_dir),
                        )
                    )
                    fold_counter += 1

        mp_ctx = multiprocessing.get_context("forkserver")
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=mp_ctx) as executor:
                [executor.submit(_noop) for _ in range(self.n_jobs)]
                model_groups: dict[str, list] = defaultdict(list)
                model_train_times: dict[str, list[float]] = defaultdict(list)

                futures = {executor.submit(_train_one_model_reg, *t): t for t in tasks}
                for future in as_completed(futures):
                    fold_number, model_id_task = futures[future][0], futures[future][1]
                    try:
                        (
                            train_idx,
                            val_idx,
                            preds,
                            model_size,
                            train_dur,
                            model_id_result,
                            fold_number,
                        ) = future.result()
                    except Exception as e:
                        raise RuntimeError(
                            f"Worker failed training {model_id_task} fold {fold_number}: {e}"
                        ) from e

                    self.log(
                        f"Trained {model_id_result} in {train_dur:.4f}s for f-{fold_number}",
                        level=2,
                        start_time=fit_start,
                    )
                    repetition = fold_number // self.k_folds
                    oof_pred_mats[model_id_result][repetition, val_idx] = preds
                    model_groups[model_id_result].append(fold_number)
                    model_train_times[model_id_result].append(train_dur)

                    if len(model_groups[model_id_result]) == expected_folds[model_id_result]:
                        del model_groups[model_id_result]
                        oof_preds[model_id_result] = np.nanmedian(
                            oof_pred_mats[model_id_result], axis=0
                        )
                        residuals = y - oof_preds[model_id_result]
                        oof_rmse = float(np.sqrt(np.nanmean(residuals**2)))
                        oof_mae = float(np.nanmean(np.abs(residuals)))
                        oof_r2 = float(r2_score(y, oof_preds[model_id_result]))
                        # Outlier-robust R² (predictions clipped to the target
                        # range before scoring) so a single off-scale sample
                        # (high-leverage ridge extrapolation) can't dominate it.
                        oof_r2_robust = _robust_r2(y, oof_preds[model_id_result])
                        base_oof_r2[model_id_result] = oof_r2
                        self._oof_scores.append(
                            {
                                "model": model_id_result,
                                "level": 0,
                                "oof_rmse": oof_rmse,
                                "oof_mae": oof_mae,
                                "oof_r2": oof_r2,
                                "oof_r2_robust": oof_r2_robust,
                                "train_time": model_train_times.pop(model_id_result),
                            }
                        )
                        self.log(
                            f"OOF  {model_id_result:<48}"
                            f"RMSE {oof_rmse:7.4f}   MAE {oof_mae:7.4f}   "
                            f"R² {oof_r2:>10.4f}   robust R² {oof_r2_robust:>8.4f}",
                            level=1,
                            start_time=fit_start,
                        )

                if not self.stacking_models:
                    return

                all_base_models = [spec.get_model_id() for spec in self.model_specs]
                if self.drop_nonpositive_r2:
                    kept = [m for m in all_base_models if base_oof_r2.get(m, 0.0) > 0.0]
                    removed = [m for m in all_base_models if m not in kept]
                    # Don't let an aggressive cut leave the stacker with nothing:
                    # if every model is non-positive, fall back to the single best.
                    if not kept:
                        best = max(all_base_models, key=lambda m: base_oof_r2.get(m, float("-inf")))
                        kept = [best]
                        removed = [m for m in all_base_models if m != best]
                        self.log(
                            "All base models have OOF R² <= 0; keeping best-R² model "
                            f"{best} ({base_oof_r2.get(best, float('nan')):.4f}) for stacking.",
                            level=1,
                        )
                    for m in removed:
                        self.log(
                            f"Stacking: REMOVED {m} (OOF R² {base_oof_r2.get(m, float('nan')):.4f} <= 0)",
                            level=1,
                        )
                    for m in kept:
                        self.log(
                            f"Stacking: KEPT    {m} (OOF R² {base_oof_r2.get(m, float('nan')):.4f})",
                            level=1,
                        )
                    self.log(
                        f"Stacking: kept {len(kept)}/{len(all_base_models)} base models "
                        f"(dropped {len(removed)} with R² <= 0)",
                        level=1,
                    )
                    self._stacking_model_order = kept
                else:
                    self._stacking_model_order = all_base_models
                oof_matrix = np.column_stack([oof_preds[mid] for mid in self._stacking_model_order])
                save_array(oof_matrix, "Xt_predictions", str(self._tmpdir))

                stacker_fold_seed = self._get_seed()
                stacker_splits = generate_folds(
                    X, y, n_splits=self.k_folds, n_repetitions=1, random_state=stacker_fold_seed,
                    stratify=False,
                )
                stack_oof_preds = {m: np.zeros(n_samples) for m in self.stacking_models}
                stack_oof_counts = {m: np.zeros(n_samples, dtype=int) for m in self.stacking_models}
                model_groups = defaultdict(list)
                model_train_times = defaultdict(list)

                stack_tasks = []
                for model_name in self.stacking_models:
                    stack_fold_rng = np.random.default_rng(self._get_seed())
                    for fold_no, (train_idx, val_idx) in enumerate(stacker_splits):
                        stack_fold_seed = int(stack_fold_rng.integers(0, 2**31 - 1))
                        stack_tasks.append(
                            (
                                fold_no,
                                model_name,
                                model_name,
                                train_idx,
                                val_idx,
                                stack_fold_seed,
                                str(self._tmpdir),
                                [FeatureSpec(feature_name="predictions")],
                                str(self._model_dir),
                            )
                        )

                futures = {executor.submit(_train_one_model_reg, *t): t for t in stack_tasks}
                for future in as_completed(futures):
                    fold_number, model_id_task = futures[future][0], futures[future][1]
                    try:
                        (
                            train_idx,
                            val_idx,
                            preds,
                            model_size,
                            train_dur,
                            model_id_result,
                            fold_number,
                        ) = future.result()
                    except Exception as e:
                        raise RuntimeError(
                            f"Worker failed stacking {model_id_task} fold {fold_number}: {e}"
                        ) from e

                    self.log(
                        f"Trained stacker {model_id_result} in {train_dur:.4f}s for f-{fold_number}",
                        level=2,
                        start_time=fit_start,
                    )
                    stack_oof_preds[model_id_result][val_idx] += preds
                    stack_oof_counts[model_id_result][val_idx] += 1
                    model_groups[model_id_result].append(fold_number)
                    model_train_times[model_id_result].append(train_dur)

                    if len(model_groups[model_id_result]) == self.k_folds:
                        del model_groups[model_id_result]
                        counts = stack_oof_counts[model_id_result]
                        avg_preds = np.where(
                            counts > 0, stack_oof_preds[model_id_result] / counts, np.nan
                        )
                        residuals = y - avg_preds
                        oof_rmse = float(np.sqrt(np.nanmean(residuals**2)))
                        oof_mae = float(np.nanmean(np.abs(residuals)))
                        oof_r2 = float(r2_score(y, avg_preds))
                        self._oof_scores.append(
                            {
                                "model": model_id_result,
                                "level": 1,
                                "oof_rmse": oof_rmse,
                                "oof_mae": oof_mae,
                                "oof_r2": oof_r2,
                                "train_time": model_train_times.pop(model_id_result),
                            }
                        )
                        self.log(
                            f"OOF  {model_id_result:<48}"
                            f"RMSE {oof_rmse:7.4f}   MAE {oof_mae:7.4f}   R² {oof_r2:>10.4f}",
                            level=1,
                            start_time=fit_start,
                        )

                self.log("Fit complete", level=1, start_time=fit_start)

        finally:
            if self._tmpdir and self._tmpdir.exists():
                shutil.rmtree(self._tmpdir)
                self._tmpdir = None

    def _predict(self, X):
        # predict_per_model computes every base model's prediction and runs the
        # stacker(s) on top; the final prediction is just the stacker output
        # (median across stacking models if there is more than one).
        per_model = self.predict_per_model(X)
        if not self.stacking_models:
            keys = [spec.get_model_id() for spec in self.model_specs]
        else:
            keys = list(self.stacking_models)
        return np.median(np.stack([per_model[k] for k in keys]), axis=0)

    def predict_per_model(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Return the test prediction of every base model, keyed by model id.

        Predictions are the median over each model's folds, the same way they feed
        the stacker. Includes models excluded from stacking (see
        ``drop_nonpositive_r2``) so their individual skill can still be measured,
        e.g. ``{m: r2_score(y_test, p) for m, p in reg.predict_per_model(X).items()}``.
        The stacking model(s) are included too (keyed by stacking model name), so
        the same dict also holds the final stacked prediction.
        """
        predict_start = perf_counter()
        features_infer = self._base_dir / "features_inference"
        features_stack = self._base_dir / "features_stack"
        os.makedirs(features_infer, exist_ok=True)

        mp_ctx = multiprocessing.get_context("forkserver")
        try:
            with ProcessPoolExecutor(max_workers=self.n_jobs, mp_context=mp_ctx) as executor:
                [executor.submit(_noop) for _ in range(self.n_jobs)]

                save_array(X, "X", str(features_infer), dtype=self._feature_dtype)
                self._compute_features(X, str(features_infer), start_time=predict_start)

                base_pred_folds = {spec.get_model_id(): [] for spec in self.model_specs}

                tasks = [
                    (
                        spec.get_model_id(),
                        str(features_infer),
                        list(spec.features),
                        str(self._model_dir),
                        fold,
                    )
                    for spec in self.model_specs
                    for fold in range(self.k_folds * spec.n_repetitions)
                ]
                futures = {executor.submit(_predict_one_model_reg, *t): t for t in tasks}
                for future in as_completed(futures):
                    model_id_task = futures[future][0]
                    try:
                        preds, predict_dur, model_id_res = future.result()
                    except Exception as e:
                        raise RuntimeError(f"Worker failed predicting {model_id_task}: {e}") from e
                    base_pred_folds[model_id_res].append(preds)

                base_preds = {
                    mid: np.median(np.stack(folds), axis=0)
                    for mid, folds in base_pred_folds.items()
                }

                if not self.stacking_models:
                    return base_preds

                # Run the stacker(s) on the base predictions and add them to the
                # returned dict, keyed by stacking model name.
                stacking_matrix = np.column_stack(
                    [base_preds[mid] for mid in self._stacking_model_order]
                )
                os.makedirs(features_stack, exist_ok=True)
                save_array(X, "X", str(features_stack), dtype=self._feature_dtype)
                save_array(stacking_matrix, "Xt_predictions", str(features_stack))

                stack_pred_folds = {m: [] for m in self.stacking_models}
                stack_tasks = [
                    (
                        model_name,
                        str(features_stack),
                        [FeatureSpec(feature_name="predictions")],
                        str(self._model_dir),
                        fold,
                    )
                    for model_name in self.stacking_models
                    for fold in range(self.k_folds)
                ]
                futures = {executor.submit(_predict_one_model_reg, *t): t for t in stack_tasks}
                for future in as_completed(futures):
                    model_id_task = futures[future][0]
                    try:
                        preds, predict_dur, model_id_res = future.result()
                    except Exception as e:
                        raise RuntimeError(
                            f"Worker failed stacking predict {model_id_task}: {e}"
                        ) from e
                    stack_pred_folds[model_id_res].append(preds)

                for m, folds in stack_pred_folds.items():
                    base_preds[m] = np.median(np.stack(folds), axis=0)
                return base_preds
        finally:
            for d in (features_infer, features_stack):
                if d.exists():
                    shutil.rmtree(d)


class SparseScaler:
    """Sparse Scaler for hydra transform (NumPy version)."""

    def __init__(self, mask=True, exponent=4):
        self.mask = mask
        self.exponent = exponent

    def _prep(self, X):
        return np.sqrt(np.clip(X, 0, None))

    def _fit_stats(self, Xt, dtype):
        # epsilon = mean((X == 0)) ** exponent + 1e-8 (bool array, no float copy needed)
        zero_frac = (Xt == 0).mean(axis=0)
        self.epsilon = zero_frac**self.exponent + 1e-8

        self.mu = Xt.mean(axis=0).astype(dtype)
        self.sigma = (Xt.std(axis=0) + self.epsilon).astype(dtype)

    def _apply(self, Xt):
        if self.mask:
            return ((Xt - self.mu) * (Xt != 0)) / self.sigma
        else:
            return (Xt - self.mu) / self.sigma

    def fit(self, X, y=None):
        self._fit_stats(self._prep(X), X.dtype)
        return self

    def transform(self, X, y=None):
        return self._apply(self._prep(X))

    def fit_transform(self, X, y=None):
        Xt = self._prep(X)
        self._fit_stats(Xt, X.dtype)
        return self._apply(Xt)


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
        select = (lambda arr: arr[idx]) if idx is not None else (lambda arr: arr)
        keys = [key for key in self.scalers_ if key in X]
        if not keys:
            return np.empty((next(iter(X.values())).shape[0], 0))

        widths = [X[key].shape[1] for key in keys]
        n_samples = select(X[keys[0]]).shape[0]
        dtype = np.result_type(*(X[key].dtype for key in keys))

        # Pre-allocate the full output once; fill it column-by-column so at most
        # one scaled feature-group chunk exists alongside it at any given time,
        # instead of holding every chunk plus a freshly hstack'd copy at once.
        out = np.empty((n_samples, sum(widths)), dtype=dtype)
        col = 0
        for key, width in zip(keys, widths):
            scaled = self.scalers_[key].transform(select(X[key]))
            out[:, col:col + width] = scaled
            del scaled
            col += width

        return out

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
