"""Tests for TSCGlueClassifier."""

import dataclasses
import tempfile

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from tscglue import utils
from tscglue.models import (
    TSCGlueClassifier,
    get_feature_transformer,
)
from tscglue.utils import FeatureSpec


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
    assert len(X_test) % 8 != 0, "batch size must leave a ragged final batch"

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0, k_folds=3, n_jobs=2, preset="medium", n_gpus=0, runs_dir=tmp_dir
        )
        model.fit(X_train, y_train)

        ref_per_model = model.predict_proba_per_model(X_test)
        ref_proba = model.predict_proba(X_test)
        ref_pred = model.predict(X_test)

        batched_per_model = model.predict_proba_per_model(X_test, predict_batch_size=8)
        batched_proba = model.predict_proba(X_test, predict_batch_size=8)
        batched_pred = model.predict(X_test, predict_batch_size=8)
        batched_labels = model.predict_per_model(X_test, predict_batch_size=8)

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
    assert np.array_equal(oversized, ref_proba)
    assert np.array_equal(batched_pred, ref_pred)


def test_predict_batch_size_from_constructor_and_per_call():
    """The constructor sets the default; a per-call value applies to that call only."""
    X_train, y_train, X_test, _ = utils.load_dataset("Coffee")

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = TSCGlueClassifier(
            random_state=0, k_folds=3, n_jobs=2, preset="medium", n_gpus=0, runs_dir=tmp_dir,
            predict_batch_size=8,
        )
        model.fit(X_train, y_train)

        from_constructor = model.predict_proba(X_test)
        from_call = model.predict_proba(X_test, predict_batch_size=8)

        # a per-call value must not be written back to the estimator
        model.predict_proba(X_test, predict_batch_size=10 * len(X_test))
        assert model.predict_batch_size == 8
        assert model._batch_for_call is None
        assert np.array_equal(model.predict_proba(X_test), from_constructor)

        with pytest.raises(ValueError):
            model.predict_proba(X_test, predict_batch_size=0)

    assert np.array_equal(from_constructor, from_call)


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


def test_best_preset_serves_meta_stacker():
    """`best` = high plus a meta stacker over the base and stacker OOF probabilities."""
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
        assert model.meta_stacking_models == ["meta-probability-et"]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        per_model = model.predict_proba_per_model(X_test)

    assert model.best_model == "meta-probability-et"
    # 12 base models + 9 stackers, one column per class each.
    assert len(model._meta_probability_columns) == (12 + 9) * 3
    np.testing.assert_allclose(proba, per_model["meta-probability-et"])
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
