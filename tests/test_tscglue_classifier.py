"""Tests for TSCGlueClassifier."""

import dataclasses
import pickle
import tempfile
import warnings

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from tscglue import models, utils
from tscglue.models import (
    MODELS_DICTIONARY,
    DictMultiScaler,
    TSCGlueClassifier,
    fold_major,
    get_feature_transformer,
    get_model_v6,
    impute_missing,
)
from tscglue.tabular import NoScaler, SparseScaler
from tscglue.utils import FeatureSpec


def test_fold_major_orders_tasks_fold_by_fold_keeping_model_order():
    # Built model by model, as in _fit: (fold, model).
    tasks = [(fold, model) for model in ["mrh", "quant", "rdst"] for fold in range(3)]

    ordered = fold_major(tasks, fold_index=0)

    assert ordered == [(fold, model) for fold in range(3) for model in ["mrh", "quant", "rdst"]]


@pytest.mark.parametrize(
    "n_jobs, n_jobs_per_model, workers, threads",
    [(12, 2, 6, 2), (12, 1, 12, 1), (5, 2, 2, 2), (1, 2, 1, 1)],
)
def test_n_jobs_per_model_splits_n_jobs_into_workers_and_threads(
    n_jobs, n_jobs_per_model, workers, threads
):
    clf = TSCGlueClassifier(n_jobs=n_jobs, n_jobs_per_model=n_jobs_per_model)

    assert clf._n_workers() == workers
    assert clf._threads_per_model() == threads


def test_n_jobs_per_model_defaults_to_2():
    assert TSCGlueClassifier().n_jobs_per_model == 2


@pytest.mark.parametrize("name", sorted(MODELS_DICTIONARY))
def test_every_fold_model_takes_n_jobs(name):
    _, clf = get_model_v6(name, seed=0, n_jobs=3)

    assert clf.n_jobs == 3


@pytest.mark.parametrize("use_idx", [False, True])
def test_dict_multi_scaler_chunked_transform_matches_whole_block(monkeypatch, tmp_path, use_idx):
    rng = np.random.default_rng(0)
    feats = {}
    for name, arr in {
        "hydra": rng.integers(0, 5, (50, 6)).astype(np.float32),
        "multirocket": rng.standard_normal((50, 9)),
        "quant": rng.standard_normal((50, 4)),
    }.items():
        np.save(tmp_path / f"{name}.npy", arr)
        feats[name] = np.load(tmp_path / f"{name}.npy", mmap_mode="r")  # as read_array does
    idx = rng.permutation(50)[:37] if use_idx else None
    scaler = DictMultiScaler(
        {"hydra": SparseScaler(), "multirocket": StandardScaler(), "quant": NoScaler()}
    ).fit(feats, idx=idx)

    # 7 rows of multirocket (9 x float64) per chunk: several chunks and a partial last one
    monkeypatch.setattr(models, "TRANSFORM_CHUNK_BYTES", 7 * 9 * 8)
    chunked = scaler.transform(feats, idx=idx)

    whole = np.hstack([
        s.transform(feats[key] if idx is None else feats[key][idx])
        for key, s in scaler.scalers_.items()
    ])
    np.testing.assert_array_equal(chunked, whole)


def test_model_accuracy_on_coffee():
    X_train, y_train, X_test, y_test = utils.load_dataset("Coffee")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(random_state=270, n_repetitions=1, k_folds=10, runs_dir=tmp_dir)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


def _make_classification_data(n_per_class=10, n_classes=15, n_timesteps=16, seed=0):
    """Synthetic multiclass series with INTEGER labels 0..n_classes-1.

    15 integer-labeled classes is what exercises the multiclass roc_auc
    label-ordering path (labels 10..14 sort differently by repr vs numerically).
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(n_classes):
        center = rng.standard_normal(n_timesteps) + c
        for _ in range(n_per_class):
            X.append((center + 0.3 * rng.standard_normal(n_timesteps))[None, :])
            y.append(c)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=int)


@pytest.mark.parametrize("eval_metric", ["accuracy", "log_loss", "roc_auc"])
def test_classifier_eval_metrics_multiclass(eval_metric):
    # 15 integer-labeled classes -> roc_auc would raise "labels must be ordered"
    # before the label-sorting fix.
    X_train, y_train = _make_classification_data(seed=0)
    X_test, y_test = _make_classification_data(seed=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0,
            n_repetitions=1,
            k_folds=3,
            n_jobs=2,
            eval_metric=eval_metric,
            runs_dir=tmp_dir,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

    assert y_pred.shape == (len(X_test),)
    assert set(np.unique(y_pred)).issubset(set(np.unique(y_train)))
    assert proba.shape == (len(X_test), len(np.unique(y_train)))
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("compute_dtype", [None, "float32", "float64"])
def test_compute_dtype(compute_dtype):
    """fit+predict works with each compute_dtype; None adopts X's own dtype.

    `low` keeps this to the four cheap representations -- the dtype plumbing is
    shared by every preset, so there is nothing extra to learn from a wider pool.
    """
    X_train, y_train, X_test, y_test = utils.load_dataset("Coffee")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=270,
            n_repetitions=1,
            k_folds=10,
            preset="low",
            compute_dtype=compute_dtype,
            runs_dir=tmp_dir,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"

    expected_dtype = np.dtype(compute_dtype) if compute_dtype else X_train.dtype
    assert model.compute_dtype == expected_dtype, (
        f"Expected compute_dtype={expected_dtype}, got {model.compute_dtype}"
    )


def test_model_on_multivariate():
    X_train, y_train, X_test, y_test = utils.load_dataset("BasicMotions")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(random_state=270, n_repetitions=1, k_folds=10, runs_dir=tmp_dir)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


@pytest.mark.parametrize("encode_labels", [False, True], ids=["string_labels", "int_labels"])
def test_label_dtype(encode_labels):
    """Test that inference works with both string and integer labels."""
    X_train, y_train, X_test, y_test = utils.load_dataset("BasicMotions")

    if encode_labels:
        labels, y_train_fit = np.unique(y_train, return_inverse=True)
        y_test_expected = np.array([np.where(labels == x)[0][0] for x in y_test])
    else:
        y_train_fit = y_train
        y_test_expected = y_test

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=270, n_repetitions=1, k_folds=10, n_jobs=1, runs_dir=tmp_dir
        )
        model.fit(X_train, y_train_fit)
        y_pred = model.predict(X_test)
        served_proba = model.predict_proba(X_test)
        proba_per_model = model.predict_proba_per_model(X_test)

    assert y_pred.shape == y_test_expected.shape
    assert served_proba.shape == (X_test.shape[0], len(model.classes_))
    assert np.isfinite(served_proba).all()
    assert proba_per_model, "no per-model probabilities returned"
    for name, proba in proba_per_model.items():
        assert proba.shape == (X_test.shape[0], len(model.classes_)), name
        assert np.isfinite(proba).all(), name

    accuracy = accuracy_score(y_test_expected, y_pred)
    assert accuracy > 0.1, f"Accuracy {accuracy} is too low (<=0.1)"
    assert accuracy <= 1.0, f"Accuracy {accuracy} is invalid (>1.0)"


# Hydra is convolved in float32, which does not reassociate identically across batch sizes,
# so batching moves hydra-fed heads by ~1e-10. Labels must still match exactly.
BATCH_PROBA_ATOL = 1e-6


def test_predict_batching_matches_unbatched():
    """A batched predict must agree with an unbatched one, head for head."""
    X_train, y_train, X_test, _ = utils.load_dataset("Coffee")
    assert len(X_test) % 15 != 0, "batch size must leave a ragged final batch"

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0, k_folds=3, n_jobs=2, preset="medium", n_gpus=0, runs_dir=tmp_dir
        )
        model.fit(X_train, y_train)

        ref_per_model = model.predict_proba_per_model(X_test)
        ref_proba = model.predict_proba(X_test)
        ref_pred = model.predict(X_test)

        batched_per_model = model.predict_proba_per_model(X_test, predict_batch_size=15)
        batched_proba = model.predict_proba(X_test, predict_batch_size=15)
        batched_pred = model.predict(X_test, predict_batch_size=15)
        batched_labels = model.predict_per_model(X_test, predict_batch_size=15)

        # a batch larger than the whole collection
        oversized = model.predict_proba(X_test, predict_batch_size=10 * len(X_test))

    assert set(batched_per_model) == set(ref_per_model)
    for name, ref in ref_per_model.items():
        got = batched_per_model[name]
        assert got.shape == ref.shape, name
        assert np.max(np.abs(got - ref)) < BATCH_PROBA_ATOL, name
        assert np.array_equal(got.argmax(axis=1), ref.argmax(axis=1)), name
        assert np.array_equal(batched_labels[name], model.classes_[ref.argmax(axis=1)]), name

    assert np.max(np.abs(batched_proba - ref_proba)) < BATCH_PROBA_ATOL
    assert np.max(np.abs(oversized - ref_proba)) < BATCH_PROBA_ATOL
    assert np.array_equal(batched_pred, ref_pred)


def test_predict_batch_size_from_constructor_and_per_call():
    """The constructor sets the default; a per-call value applies to that call only."""
    X_train, y_train, X_test, _ = utils.load_dataset("Coffee")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0, k_folds=3, n_jobs=2, preset="medium", n_gpus=0, runs_dir=tmp_dir,
            predict_batch_size=15,
        )
        model.fit(X_train, y_train)

        from_constructor = model.predict_proba(X_test)
        from_call = model.predict_proba(X_test, predict_batch_size=15)

        # a per-call value must not be written back to the estimator
        model.predict_proba(X_test, predict_batch_size=10 * len(X_test))
        assert model.predict_batch_size == 15
        assert model._batch_for_call is None
        assert np.max(np.abs(model.predict_proba(X_test) - from_constructor)) < BATCH_PROBA_ATOL

        with pytest.raises(ValueError):
            model.predict_proba(X_test, predict_batch_size=0)

    assert np.max(np.abs(from_constructor - from_call)) < BATCH_PROBA_ATOL


def test_predict_batching_on_fallback_path():
    """A singleton class forces the fallback, which has its own batching loop."""
    X_train, y_train = _make_classification_data(n_per_class=4, n_classes=3, seed=0)
    X_test, _ = _make_classification_data(n_per_class=4, n_classes=3, seed=1)
    y_train[-1] = 99  # singleton class -> fold training impossible -> fallback

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0, k_folds=3, n_jobs=2, preset="medium", n_gpus=0, runs_dir=tmp_dir
        )
        model.fit(X_train, y_train)
        assert model._fallback_path.exists(), "expected the fallback to be fitted"

        ref_proba = model.predict_proba(X_test)
        ref_pred = model.predict(X_test)
        batched_proba = model.predict_proba(X_test, predict_batch_size=8)
        batched_pred = model.predict(X_test, predict_batch_size=8)

    assert np.max(np.abs(batched_proba - ref_proba)) < BATCH_PROBA_ATOL
    assert np.array_equal(batched_pred, ref_pred)


def test_high_preset_shares_feature_specs():
    """Both heads of each representation must reuse the identical cached feature (same seed)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(random_state=0, preset="high", runs_dir=tmp_dir)

    assert len(model.model_specs) == 12
    # 12 models but only 8 transforms: multirocket, hydra, quant, rdst,
    # rstsf-random, mantis, chronos2, weasel
    assert len(model.features_list) == 8

    specs = {spec.model_name: spec for spec in model.model_specs}
    for ridge_name, etc_name in [
        ("multirockethydra-bestk-p-ridgecv", "multirockethydra-etc"),
        ("quant-p-ridgecv", "quant-etc"),
        ("rdst-p-ridgecv", "rdst-etc"),
        ("rstsf-random-p-ridgecv", "rstsf-random-etc"),
        ("fm-p-ridgecv", "fm-etc"),
        ("weasel-bestk-p-ridgecv", "weasel-etc"),
    ]:
        ridge_ids = [ft.get_feature_id() for ft in specs[ridge_name].features]
        etc_ids = [ft.get_feature_id() for ft in specs[etc_name].features]
        assert ridge_ids == etc_ids, f"{ridge_name} and {etc_name} do not share features"


def test_high_preset_fit_predict():
    X_train, y_train = _make_classification_data(seed=0)
    X_test, _ = _make_classification_data(seed=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0, n_repetitions=1, k_folds=3, n_jobs=2, preset="high", runs_dir=tmp_dir
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)

    assert y_pred.shape == (len(X_test),)
    assert set(np.unique(y_pred)).issubset(set(np.unique(y_train)))
    assert proba.shape == (len(X_test), len(np.unique(y_train)))
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    ("eval_metric", "expected_head"),
    [("accuracy", "probability-ridgecv"), ("log_loss", "probability-et")],
)
def test_low_preset_serves_its_single_stacker(eval_metric, expected_head):
    """`low` trains exactly one stacker, picked pre-fit by eval_metric, and serves it."""
    X_train, y_train = _make_classification_data(n_per_class=6, n_classes=3, seed=0)
    X_test, _ = _make_classification_data(n_per_class=2, n_classes=3, seed=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0,
            k_folds=3,
            n_jobs=2,
            preset="low",
            eval_metric=eval_metric,
            runs_dir=tmp_dir,
        )
        assert model.stacking_models == [expected_head]
        # low drops fm, so no foundation-model features are built.
        assert len(model.model_specs) == 4

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)

    assert model.best_model == expected_head
    assert proba.shape == (len(X_test), len(np.unique(y_train)))
    assert np.isfinite(proba).all()


def test_best_preset_serves_blend_head():
    """`best` = high, served through the wmean-T-ll-L2 blend of base and stacker probabilities."""
    X_train, y_train = _make_classification_data(n_per_class=9, n_classes=3, seed=0)
    X_test, _ = _make_classification_data(n_per_class=2, n_classes=3, seed=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0,
            k_folds=3,
            n_jobs=2,
            preset="best",
            eval_metric="log_loss",
            runs_dir=tmp_dir,
        )
        assert len(model.model_specs) == 12
        assert len(model.stacking_models) == 9

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        per_model = model.predict_proba_per_model(X_test)

    assert model.best_model == "wmean-T-ll-L2"
    # 12 base models + 9 stackers, and nothing above them.
    assert len(model.blend_members_) == 12 + 9
    assert set(per_model) == set(model.blend_members_)
    members = np.stack([per_model[m] for m in model.blend_members_])
    np.testing.assert_allclose(proba, model.blend_.predict(members))
    assert proba.shape == (len(X_test), 3)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_missing_classes_helper():
    """_missing_classes reports labels of y that a fitted model cannot predict."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(random_state=0, runs_dir=tmp_dir)

    model.classes_ = np.array(["a", "b", "c"])
    assert model._missing_classes(np.array(["a", "b", "c"])) == []
    assert model._missing_classes(np.array(["a", "c"])) == ["b"]
    assert model._missing_classes(np.array(["c"])) == ["a", "b"]

    # Integer labels: comparison is on str, so dtype differences must not matter.
    model.classes_ = np.array([0, 1, 2])
    assert model._missing_classes(np.array([0, 1, 2], dtype=np.int32)) == []
    assert model._missing_classes(np.array([0, 2])) == [1]


def test_default_fold_method_is_round_robin():
    y = np.repeat(np.arange(4), 12)
    X = np.zeros((len(y), 1))
    expected = utils.get_folds(X, y, n_splits=10, random_state=3, fold_method="round-robin")
    assert utils.get_folds(X, y, n_splits=10, random_state=3) == expected
    with tempfile.TemporaryDirectory() as tmp_dir:
        assert TSCGlueClassifier(runs_dir=tmp_dir).fold_method == "round-robin"


def test_stratified_kfold_fold_method_matches_sklearn():
    """fold_method="stratified-kfold" keeps StratifiedKFold's splits."""
    from sklearn.model_selection import StratifiedKFold

    y = np.repeat(np.arange(4), 12)
    X = np.zeros((len(y), 1))
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=3)
    expected = [(tr.tolist(), va.tolist()) for tr, va in skf.split(X, y)]
    explicit = utils.get_folds(X, y, n_splits=10, random_state=3, fold_method="stratified-kfold")
    assert explicit == expected


def test_round_robin_folds_keep_every_class():
    """52 classes x 2 cases, where StratifiedKFold refuses: round-robin folds partition the
    cases, differ in size by at most one, and keep every class in every training part."""
    y = np.repeat(np.arange(52), 2)
    X = np.zeros((len(y), 1))
    folds = utils.get_folds(X, y, n_splits=10, random_state=0, fold_method="round-robin")

    assert len(folds) == 10
    assert sorted(i for _, va in folds for i in va) == list(range(len(y)))
    sizes = [len(va) for _, va in folds]
    assert max(sizes) - min(sizes) <= 1
    for tr, va in folds:
        assert set(tr).isdisjoint(va)
        assert len(np.unique(y[tr])) == 52
    assert folds == utils.get_folds(X, y, n_splits=10, random_state=0, fold_method="round-robin")
    assert folds != utils.get_folds(X, y, n_splits=10, random_state=1, fold_method="round-robin")


def test_unknown_fold_method_raises():
    with pytest.raises(ValueError):
        utils.get_folds(np.zeros((4, 1)), np.array([0, 0, 1, 1]), n_splits=2, fold_method="kfold")
    with tempfile.TemporaryDirectory() as tmp_dir, pytest.raises(AssertionError):
        TSCGlueClassifier(fold_method="kfold", runs_dir=tmp_dir)


def test_round_robin_fold_method_fits_the_stack():
    """2 cases per class and k_folds=3: round-robin keeps every class in every training
    part, so the stack is fitted instead of the fallback."""
    X_train, y_train = _make_classification_data(n_per_class=2, n_classes=6, seed=0)
    X_test, _ = _make_classification_data(n_per_class=2, n_classes=6, seed=1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0, k_folds=3, n_jobs=2, preset="low", n_gpus=0, runs_dir=tmp_dir,
            fold_method="round-robin",
        )
        model.fit(X_train, y_train)
        assert not model._fallback_path.exists(), "round-robin folds should not need the fallback"
        assert model.predict_proba(X_test).shape == (len(X_test), 6)


def test_hydra_transformer_is_device_routed():
    """A non-cpu device swaps aeon's hydra for the torch one; cpu keeps aeon's."""
    from aeon.transformations.collection.convolution_based._hydra import HydraTransformer

    from tscglue.features_gpu import HydraTransformerDevice

    assert isinstance(get_feature_transformer("hydra", seed=0), HydraTransformer)
    assert isinstance(get_feature_transformer("hydra", seed=0, device="cpu"), HydraTransformer)

    gpu = get_feature_transformer("hydra", seed=0, device="cuda")
    assert isinstance(gpu, HydraTransformerDevice)
    assert gpu.device == "cuda"
    assert gpu.random_state == 0


def _lane_names(model):
    """Effective lanes, mirroring the callers: no device means no split at all."""
    if model.n_gpus == 0:
        return [], [ft.feature_name for ft in model.features_list]
    gpu_features, cpu_features = FeatureSpec.split_lanes(model.features_list)
    return [ft.feature_name for ft in gpu_features], [ft.feature_name for ft in cpu_features]


def _spec(model, name):
    return next(ft for ft in model.features_list if ft.feature_name == name)


def test_hydra_goes_on_the_gpu_lane_first():
    """With a GPU, hydra joins the device lane ahead of the foundation models."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(random_state=0, n_gpus=1, preset="high", runs_dir=tmp_dir)

        gpu_lane, cpu_lane = _lane_names(model)

        # hydra leads, so it releases its device memory before mantis/chronos2 allocate.
        assert gpu_lane == ["hydra", "mantis", "chronos2"]
        assert "hydra" not in cpu_lane
        assert model._feature_device(_spec(model, "hydra")) == "cuda"

        # Only support_gpu features move -- every other one is still built for the cpu.
        assert model._feature_device(_spec(model, "quant")) == "cpu"
        assert model._feature_device(_spec(model, "multirocket")) == "cpu"
        assert model._feature_device(_spec(model, "rdst")) == "cpu"
        assert model._feature_device(_spec(model, "mantis")) == "cuda"


def test_no_device_lane_without_a_gpu():
    """n_gpus=0 empties the device lane: every feature is built for the cpu."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(random_state=0, n_gpus=0, preset="high", runs_dir=tmp_dir)

        gpu_lane, cpu_lane = _lane_names(model)

        assert gpu_lane == []
        assert "hydra" in cpu_lane
        for feature in ["hydra", "mantis", "chronos2", "quant", "multirocket"]:
            assert model._feature_device(_spec(model, feature)) == "cpu"


def test_support_gpu_flag_steers_the_lane():
    """The lane split follows FeatureSpec.support_gpu, nothing else."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(random_state=0, n_gpus=1, preset="high", runs_dir=tmp_dir)

        model.features_list = [
            dataclasses.replace(ft, support_gpu=ft.feature_name == "quant")
            for ft in model.features_list
        ]

        gpu_lane, cpu_lane = _lane_names(model)
        assert gpu_lane == ["quant"]
        assert "hydra" in cpu_lane
        assert model._feature_device(_spec(model, "quant")) == "cuda"
        assert model._feature_device(_spec(model, "hydra")) == "cpu"


def test_hydra_device_falls_back_to_cpu_without_cuda():
    """A cuda-fitted transformer stays usable where there is no cuda to unpickle onto."""
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("guard only fires when CUDA is absent")

    from tscglue.features_gpu import HydraTransformerDevice

    X, _ = _make_classification_data(n_per_class=3, n_classes=2, n_timesteps=64)

    transformer = HydraTransformerDevice(random_state=0, device="cuda").fit(X)
    with pytest.warns(RuntimeWarning, match="no CUDA device"):
        Xt = transformer.transform(X)

    # Kernels are drawn on the cpu, so falling back must reproduce the cpu run exactly.
    expected = HydraTransformerDevice(random_state=0, device="cpu").fit_transform(X)
    np.testing.assert_array_equal(Xt, expected)
    assert transformer.device == "cuda", "device is a constructor param and must not be mutated"


# ----------------- missing values -----------------


def test_impute_missing_fills_interior_edges_and_empty_series():
    nan = np.nan
    X = np.array(
        [
            [[1.0, nan, 3.0, nan, nan, 6.0], [nan, nan, 2.0, 4.0, nan, nan]],
            [[nan, nan, nan, nan, nan, nan], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
        ]
    )
    out = impute_missing(X)
    np.testing.assert_array_equal(out[0, 0], [1, 2, 3, 4, 5, 6])  # interior gaps
    np.testing.assert_array_equal(out[0, 1], [2, 2, 2, 4, 4, 4])  # leading / trailing gaps
    np.testing.assert_array_equal(out[1, 0], np.zeros(6))  # nothing observed
    np.testing.assert_array_equal(out[1, 1], X[1, 1])  # untouched
    assert np.isnan(X).sum() == 13, "the input must not be modified"
    assert out.dtype == X.dtype


def test_impute_missing_returns_input_without_nans():
    X = np.random.default_rng(0).standard_normal((3, 2, 8))
    assert impute_missing(X) is X


def _nan_at(X, positions):
    X = X.copy()
    for i, c, t in positions:
        X[i, c, t] = np.nan
    return X


def test_prepare_x_masks_channels_with_nans_in_train():
    rng = np.random.default_rng(0)
    X = _nan_at(rng.standard_normal((5, 4, 10)), [(0, 0, 3), (2, 3, 0), (4, 3, 9)])
    clf = TSCGlueClassifier()
    Xp = clf._prepare_x(X, fit=True)

    assert clf.mask_channels_ == [0, 3]
    assert Xp.shape == (5, 4 + 2, 10)
    assert not np.isnan(Xp).any()
    np.testing.assert_array_equal(Xp[:, :4], impute_missing(X))
    np.testing.assert_array_equal(Xp[:, 4], (~np.isnan(X[:, 0])).astype(float))
    np.testing.assert_array_equal(Xp[:, 5], (~np.isnan(X[:, 3])).astype(float))
    assert Xp[0, 4, 3] == 0 and Xp[2, 5, 0] == 0 and Xp[4, 5, 9] == 0
    assert Xp[:, 4:].sum() == 2 * 5 * 10 - 3, "1 = observed, 0 = imputed"


def test_prepare_x_recreates_masks_at_predict_and_warns_on_new_nan_channels():
    rng = np.random.default_rng(0)
    clf = TSCGlueClassifier()
    clf._prepare_x(_nan_at(rng.standard_normal((5, 4, 10)), [(0, 0, 3), (1, 3, 5)]), fit=True)
    assert clf.mask_channels_ == [0, 3]

    # predict: NaNs only in channel 2, which had none during fit
    X_test = _nan_at(rng.standard_normal((3, 4, 10)), [(1, 2, 4)])
    with pytest.warns(UserWarning, match=r"Channels \[2\]"):
        Xp = clf._prepare_x(X_test)

    assert Xp.shape == (3, 4 + 2, 10), "same layout as in training"
    assert not np.isnan(Xp).any()
    np.testing.assert_array_equal(Xp[:, :4], impute_missing(X_test))
    np.testing.assert_array_equal(Xp[:, 4:], 1.0)  # channels 0 and 3 are fully observed


def test_prepare_x_masks_follow_the_test_data():
    rng = np.random.default_rng(0)
    clf = TSCGlueClassifier()
    clf._prepare_x(_nan_at(rng.standard_normal((5, 2, 10)), [(0, 1, 3)]), fit=True)

    X_test = _nan_at(rng.standard_normal((3, 2, 10)), [(2, 1, 7)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # channel 1 is masked, so no warning
        Xp = clf._prepare_x(X_test)
    assert Xp.shape == (3, 3, 10)
    assert Xp[2, 2, 7] == 0 and Xp[:, 2].sum() == 3 * 10 - 1


def test_prepare_x_without_nans_in_train_adds_no_mask():
    rng = np.random.default_rng(0)
    clf = TSCGlueClassifier()
    X = rng.standard_normal((5, 2, 10))
    assert clf._prepare_x(X, fit=True) is X
    assert clf.mask_channels_ == []

    with pytest.warns(UserWarning, match=r"Channels \[0\]"):
        Xp = clf._prepare_x(_nan_at(X, [(0, 0, 0)]))
    assert Xp.shape == X.shape and not np.isnan(Xp).any()


def test_prepare_x_impute_mode_never_masks():
    rng = np.random.default_rng(0)
    clf = TSCGlueClassifier(missing_values="impute")
    X = _nan_at(rng.standard_normal((5, 2, 10)), [(0, 1, 3)])
    Xp = clf._prepare_x(X, fit=True)
    assert clf.mask_channels_ == []
    assert Xp.shape == X.shape and not np.isnan(Xp).any()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        clf._prepare_x(_nan_at(X, [(1, 0, 2)]))


def test_missing_values_error_mode_raises():
    X, y = _make_classification_data(n_per_class=4, n_classes=2)
    X = _nan_at(X, [(0, 0, 3)])
    with tempfile.TemporaryDirectory() as tmp_dir:
        clf = TSCGlueClassifier(missing_values="error", runs_dir=tmp_dir)
        with pytest.raises(ValueError, match="missing_values='error'"):
            clf.fit(X, y)


def test_unknown_missing_values_mode_raises():
    with pytest.raises(AssertionError, match="missing_values"):
        TSCGlueClassifier(missing_values="drop")


def test_mask_channels_survive_pickling():
    rng = np.random.default_rng(0)
    clf = TSCGlueClassifier()
    clf._prepare_x(_nan_at(rng.standard_normal((5, 3, 10)), [(0, 2, 1)]), fit=True)
    restored = pickle.loads(pickle.dumps(clf))
    assert restored.mask_channels_ == [2]
    assert restored.missing_values == "impute_mask"


def _two_channel_data(seed):
    X, y = _make_classification_data(n_per_class=10, n_classes=3, n_timesteps=32, seed=seed)
    noise = np.random.default_rng(seed).standard_normal(X.shape).astype(X.dtype)
    return np.concatenate([X, noise], axis=1), y


@pytest.mark.parametrize("mode, mask_channels", [("impute_mask", [0]), ("impute", [])])
def test_fit_predict_with_missing_values(mode, mask_channels):
    """End to end: NaNs in channel 0 at fit; channels 0 and 1 at predict."""
    X_train, y_train = _two_channel_data(seed=0)
    X_test, _ = _two_channel_data(seed=1)
    rng = np.random.default_rng(0)
    X_train[:, 0][rng.random(X_train[:, 0].shape) < 0.2] = np.nan
    X_test[rng.random(X_test.shape) < 0.2] = np.nan

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0,
            k_folds=3,
            n_jobs=2,
            preset="low",
            missing_values=mode,
            runs_dir=tmp_dir,
        )
        model.fit(X_train, y_train)
        assert model.mask_channels_ == mask_channels
        assert not model._fallback_path.exists(), "the stack must run, not the fallback"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            per_model = model.predict_proba_per_model(X_test)
        new_channel = [w for w in caught if "Channels [1]" in str(w.message)]
        assert bool(new_channel) == (mode == "impute_mask")

    assert proba.shape == (len(X_test), 3)
    assert np.isfinite(proba).all()
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert y_pred.shape == (len(X_test),)
    assert per_model and all(np.isfinite(p).all() for p in per_model.values())


def test_clean_data_gives_the_same_predictions_in_every_mode():
    """Without NaNs, missing_values must not change anything."""
    X_train, y_train = _two_channel_data(seed=0)
    X_test, _ = _two_channel_data(seed=1)
    probas = {}
    for mode in ("impute_mask", "error"):
        with tempfile.TemporaryDirectory() as tmp_dir:
            model = TSCGlueClassifier(
                random_state=0,
                k_folds=3,
                n_jobs=2,
                preset="low",
                missing_values=mode,
                runs_dir=tmp_dir,
            )
            model.fit(X_train, y_train)
            assert model.mask_channels_ == []
            probas[mode] = model.predict_proba(X_test)
    assert np.max(np.abs(probas["impute_mask"] - probas["error"])) < BATCH_PROBA_ATOL
