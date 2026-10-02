"""Regression stacking model and its regression-only helpers."""

import multiprocessing
import os
import shutil
import uuid
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
from aeon.regression.base import BaseRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tscglue.models import (
    DictMultiScaler,
    _fit_transform_in_subprocess,
    _fit_transform_inline,
    _load_feature_dict_v10,
    _transform_in_subprocess,
    _transform_inline,
    generate_folds,
)
from tscglue.utils import FeatureSpec, ModelSpec, read_array, read_model, save_array, save_model
from tscglue.tabular import ClippedRegressor, NoScaler, SparseScaler
from tscglue.utils import _noop, _run_in_subprocess, log


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
        use_subprocess = feature_name not in ("multirocket", "rdst", "rstsf-random")
        support_gpu = feature_name in ("hydra", "mantis", "chronos2")
        if feature_name in ("quant", "mantis", "chronos2", "tsfresh"):
            return FeatureSpec(
                feature_name=feature_name,
                use_subprocess=use_subprocess,
                support_gpu=support_gpu,
            )
        return FeatureSpec(
            feature_name=feature_name,
            feature_seed=int(group_rng.integers(0, 2**31 - 1)),
            use_subprocess=use_subprocess,
            support_gpu=support_gpu,
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
        self._compute_dtype: np.dtype | None = None

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
                        self._compute_dtype,
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
                    self._compute_dtype,
                )
            Xt = read_array(f"Xt_{ft.get_feature_id()}", directory)
            elapsed = perf_counter() - t0
            log(
                f"Fit+transformed {ft.get_feature_id()} features {Xt.shape} ({Xt.nbytes / (1024 * 1024):.2f} MB) dtype={Xt.dtype} in {elapsed:.4f}s",
                level=1,
                start_time=fit_start_time,
                verbose=self.verbose,
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
            t0 = perf_counter()
            if ft.use_subprocess:
                _run_in_subprocess(
                    _transform_in_subprocess,
                    (
                        ft.get_feature_id(),
                        X_path,
                        str(self._model_dir),
                        directory,
                        self._compute_dtype,
                        self.verbose,
                    ),
                )
            else:
                _transform_inline(
                    ft.get_feature_id(),
                    X_path,
                    str(self._model_dir),
                    directory,
                    self._compute_dtype,
                )
            Xt = read_array(f"Xt_{ft.get_feature_id()}", directory)
            log(
                f"Computed {ft.get_feature_id()} features {Xt.shape} in {perf_counter() - t0:.4f}s",
                level=1,
                start_time=start_time,
                verbose=self.verbose,
            )

    def _fit(self, X, y):
        fit_start = perf_counter()
        self._compute_dtype = np.asarray(X).dtype

        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._tmpdir, exist_ok=True)
        save_array(X, "X", str(self._tmpdir), dtype=self._compute_dtype)
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

                    log(
                        f"Trained {model_id_result} in {train_dur:.4f}s for f-{fold_number}",
                        level=2,
                        start_time=fit_start,
                        verbose=self.verbose,
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
                        base_oof_r2[model_id_result] = oof_r2
                        self._oof_scores.append(
                            {
                                "model": model_id_result,
                                "level": 0,
                                "oof_rmse": oof_rmse,
                                "oof_mae": oof_mae,
                                "oof_r2": oof_r2,
                                "train_time": model_train_times.pop(model_id_result),
                            }
                        )
                        log(
                            f"OOF  {model_id_result:<48}"
                            f"RMSE {oof_rmse:7.4f}   MAE {oof_mae:7.4f}   R² {oof_r2:>10.4f}",
                            level=1,
                            start_time=fit_start,
                            verbose=self.verbose,
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
                        log(
                            "All base models have OOF R² <= 0; keeping best-R² model "
                            f"{best} ({base_oof_r2.get(best, float('nan')):.4f}) for stacking.",
                            level=1,
                            verbose=self.verbose,
                        )
                    for m in removed:
                        log(
                            f"Stacking: REMOVED {m} (OOF R² {base_oof_r2.get(m, float('nan')):.4f} <= 0)",
                            level=1,
                            verbose=self.verbose,
                        )
                    for m in kept:
                        log(
                            f"Stacking: KEPT    {m} (OOF R² {base_oof_r2.get(m, float('nan')):.4f})",
                            level=1,
                            verbose=self.verbose,
                        )
                    log(
                        f"Stacking: kept {len(kept)}/{len(all_base_models)} base models "
                        f"(dropped {len(removed)} with R² <= 0)",
                        level=1,
                        verbose=self.verbose,
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

                    log(
                        f"Trained stacker {model_id_result} in {train_dur:.4f}s for f-{fold_number}",
                        level=2,
                        start_time=fit_start,
                        verbose=self.verbose,
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
                        log(
                            f"OOF  {model_id_result:<48}"
                            f"RMSE {oof_rmse:7.4f}   MAE {oof_mae:7.4f}   R² {oof_r2:>10.4f}",
                            level=1,
                            start_time=fit_start,
                            verbose=self.verbose,
                        )

                log("Fit complete", level=1, start_time=fit_start, verbose=self.verbose)

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

                save_array(X, "X", str(features_infer), dtype=self._compute_dtype)
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
                save_array(X, "X", str(features_stack), dtype=self._compute_dtype)
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
