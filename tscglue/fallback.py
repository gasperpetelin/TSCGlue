"""Standalone feature-pipeline baselines built from TSCGlue's representations.

Each model is a plain aeon classifier: extract features with one or more of the
transformers TSCGlue already uses, concatenate the blocks, and fit a single
sklearn head. No fold training, no stacking — these serve as independent
per-metric baselines and as strong probabilistic fallback candidates.

Concrete models (bake-off precedents from "Bake off redux", Table 14):

- ``QuantET``          quant -> ExtraTrees                        ~ QUANT (NLL 0.497, AUROC 0.962)
- ``MultiET``          quant + rstsf-random + rdst -> ExtraTrees  ~ RIST (AUROC 0.966)
- ``MRHydraET``        multirocket + hydra -> ExtraTrees          rocket features, forest probabilities
- ``ShapeDictET``      rdst + weasel -> ExtraTrees                shapelet + dictionary domains
- ``AllFeaturesET``    all six representations -> ExtraTrees      kitchen-sink forest
- ``MRHydraLogistic``  multirocket + hydra -> LogisticCV          MR-Hydra accuracy, real probabilities
- ``MRHydraRidge``     multirocket + hydra -> bestk ridge         ~ MR-Hydra (ACC 0.884)
- ``AllFeaturesRidge`` multirocket + hydra + rdst + weasel -> bestk ridge  top accuracy features combined
"""

import numpy as np
from aeon.classification.base import BaseClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

from tscglue.models import (
    AutoSelectKBestClassifier,
    RareClassSafeLogisticCV,
    SparseScaler,
    get_feature_transformer,
)


class TSCFeatureBaseline(BaseClassifier):
    """Concatenated-feature pipeline classifier over TSCGlue's transformers.

    Parameters
    ----------
    features : tuple of str
        Feature transformer names understood by ``get_feature_transformer``
        (e.g. "quant", "multirocket", "hydra", "rdst", "rstsf-random", "weasel").
    head : str
        "et" (ExtraTrees), "logistic" (RareClassSafeLogisticCV) or
        "ridge" (AutoSelectKBestClassifier with RidgeClassifierCVDecisionProba).
    """

    _tags = {"capability:multivariate": True}

    PROBA_EPS = 1e-4

    def __init__(
        self,
        features=("quant",),
        head="et",
        n_estimators=500,
        random_state=None,
        n_jobs=1,
        verbose=0,
    ):
        super().__init__()
        self.features = features
        self.head = head
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ----------------- components -----------------

    def _make_head(self, seed: int):
        if self.head == "et":
            return ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                criterion="entropy",
                max_features="sqrt",
                random_state=seed,
                n_jobs=self.n_jobs,
            )
        if self.head == "logistic":
            return RareClassSafeLogisticCV()
        if self.head == "ridge":
            return AutoSelectKBestClassifier()
        raise ValueError(f"Unknown head {self.head!r}; expected 'et', 'logistic' or 'ridge'")

    def _block_scaler(self, feature_name: str):
        # Tree heads take raw features; linear heads get the same per-block
        # scaling the stacked pipeline uses (SparseScaler for hydra counts).
        if self.head == "et":
            return None
        if feature_name == "hydra":
            return SparseScaler()
        return StandardScaler()

    def _feature_input(self, feature_name: str, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float64) if feature_name == "rdst" else X

    # ----------------- aeon API -----------------

    def _fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        self.transformers_ = {}
        self.scalers_ = {}
        blocks = []
        for name in self.features:
            seed = int(rng.integers(0, 2**31 - 1))
            transformer = get_feature_transformer(name, seed=seed, n_jobs=self.n_jobs)
            Xt = np.asarray(
                transformer.fit_transform(self._feature_input(name, X)), dtype=np.float32
            )
            self.transformers_[name] = transformer
            scaler = self._block_scaler(name)
            if scaler is not None:
                Xt = np.asarray(scaler.fit_transform(Xt), dtype=np.float32)
                self.scalers_[name] = scaler
            if self.verbose:
                print(f"[{type(self).__name__}] {name}: {Xt.shape[1]} features")
            blocks.append(Xt)
        Xt_all = np.hstack(blocks)
        self.head_ = self._make_head(int(rng.integers(0, 2**31 - 1)))
        self.head_.fit(Xt_all, y)
        return self

    def _transform_blocks(self, X) -> np.ndarray:
        blocks = []
        for name in self.features:
            Xt = np.asarray(
                self.transformers_[name].transform(self._feature_input(name, X)),
                dtype=np.float32,
            )
            if name in self.scalers_:
                Xt = np.asarray(self.scalers_[name].transform(Xt), dtype=np.float32)
            blocks.append(Xt)
        return np.hstack(blocks)

    def _predict_proba(self, X):
        proba = self.head_.predict_proba(self._transform_blocks(X))
        head_classes = np.asarray(self.head_.classes_)
        if not np.array_equal(head_classes, self.classes_):
            order = [int(np.where(head_classes == c)[0][0]) for c in self.classes_]
            proba = proba[:, order]
        proba = np.clip(proba, self.PROBA_EPS, None)
        return proba / proba.sum(axis=1, keepdims=True)

    def _predict(self, X):
        return self.classes_[np.argmax(self._predict_proba(X), axis=1)]


# ----------------- forest-head models (log_loss / roc_auc) -----------------


class QuantET(TSCFeatureBaseline):
    """quant -> ExtraTrees. In-house QUANT twin: 2nd-best NLL and AUROC in the bake-off."""

    def __init__(self, n_estimators=500, random_state=None, n_jobs=1, verbose=0):
        super().__init__(
            features=("quant",), head="et", n_estimators=n_estimators,
            random_state=random_state, n_jobs=n_jobs, verbose=verbose,
        )


class MultiET(TSCFeatureBaseline):
    """quant + rstsf-random + rdst -> ExtraTrees. RIST-style multi-domain forest (AUROC 0.966 precedent)."""

    def __init__(self, n_estimators=500, random_state=None, n_jobs=1, verbose=0):
        super().__init__(
            features=("quant", "rstsf-random", "rdst"), head="et", n_estimators=n_estimators,
            random_state=random_state, n_jobs=n_jobs, verbose=verbose,
        )


class MRHydraET(TSCFeatureBaseline):
    """multirocket + hydra -> ExtraTrees. Rocket features with forest probabilities instead of ridge."""

    def __init__(self, n_estimators=500, random_state=None, n_jobs=1, verbose=0):
        super().__init__(
            features=("multirocket", "hydra"), head="et", n_estimators=n_estimators,
            random_state=random_state, n_jobs=n_jobs, verbose=verbose,
        )


class ShapeDictET(TSCFeatureBaseline):
    """rdst + weasel -> ExtraTrees. Shapelet + dictionary domains under one forest."""

    def __init__(self, n_estimators=500, random_state=None, n_jobs=1, verbose=0):
        super().__init__(
            features=("rdst", "weasel"), head="et", n_estimators=n_estimators,
            random_state=random_state, n_jobs=n_jobs, verbose=verbose,
        )


class AllFeaturesET(TSCFeatureBaseline):
    """All six CPU representations -> one ExtraTrees. Kitchen-sink maximum-diversity forest."""

    def __init__(self, n_estimators=500, random_state=None, n_jobs=1, verbose=0):
        super().__init__(
            features=("multirocket", "hydra", "quant", "rstsf-random", "rdst", "weasel"),
            head="et", n_estimators=n_estimators,
            random_state=random_state, n_jobs=n_jobs, verbose=verbose,
        )


# ----------------- linear-head models (accuracy) -----------------


class MRHydraLogistic(TSCFeatureBaseline):
    """multirocket + hydra -> LogisticCV. MR-Hydra-level features with calibrated probabilities."""

    def __init__(self, random_state=None, n_jobs=1, verbose=0):
        super().__init__(
            features=("multirocket", "hydra"), head="logistic",
            random_state=random_state, n_jobs=n_jobs, verbose=verbose,
        )


class MRHydraRidge(TSCFeatureBaseline):
    """multirocket + hydra -> bestk ridge. In-house MR-Hydra twin (accuracy anchor)."""

    def __init__(self, random_state=None, n_jobs=1, verbose=0):
        super().__init__(
            features=("multirocket", "hydra"), head="ridge",
            random_state=random_state, n_jobs=n_jobs, verbose=verbose,
        )


class AllFeaturesRidge(TSCFeatureBaseline):
    """multirocket + hydra + rdst + weasel -> bestk ridge. Union of the bake-off's top accuracy features."""

    def __init__(self, random_state=None, n_jobs=1, verbose=0):
        super().__init__(
            features=("multirocket", "hydra", "rdst", "weasel"), head="ridge",
            random_state=random_state, n_jobs=n_jobs, verbose=verbose,
        )


BASELINES = {
    cls.__name__: cls
    for cls in (
        QuantET,
        MultiET,
        MRHydraET,
        ShapeDictET,
        AllFeaturesET,
        MRHydraLogistic,
        MRHydraRidge,
        AllFeaturesRidge,
    )
}
