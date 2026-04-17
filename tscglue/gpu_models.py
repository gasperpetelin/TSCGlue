import time

import numpy as np
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from aeon.classification import BaseClassifier
from aeon.classification.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.utils.validation import check_n_jobs


class MRHydraClassifier(BaseClassifier):
    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        n_kernels: int = 8,
        n_groups: int = 64,
        estimator=None,
        n_jobs: int = 1,
        random_state=None,
    ):
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.random_state = random_state

        super().__init__()

    def _fit(self, X, y):
        self._n_jobs = check_n_jobs(self.n_jobs)

        t0 = time.perf_counter()
        self._transform_hydra = HydraTransformer(
            n_kernels=self.n_kernels,
            n_groups=self.n_groups,
            n_jobs=self._n_jobs,
            random_state=self.random_state,
        )
        Xt_hydra = self._transform_hydra.fit_transform(X)
        self._scale_hydra = _SparseScaler()
        Xt_hydra = self._scale_hydra.fit_transform(Xt_hydra)
        self.hydra_time_ = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._transform_multirocket = MultiRocket(
            n_jobs=self._n_jobs,
            random_state=self.random_state,
        )
        Xt_multirocket = self._transform_multirocket.fit_transform(X)
        self._scale_multirocket = StandardScaler()
        Xt_multirocket = self._scale_multirocket.fit_transform(Xt_multirocket)
        self.multirocket_time_ = time.perf_counter() - t0

        Xt = np.concatenate((Xt_hydra, Xt_multirocket), axis=1)

        if self.estimator is None:
            self.classifier_ = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        else:
            self.classifier_ = self.estimator

        t0 = time.perf_counter()
        self.classifier_.fit(Xt, y)
        self.classifier_time_ = time.perf_counter() - t0

        return self

    def _predict(self, X) -> np.ndarray:
        Xt_hydra = self._transform_hydra.transform(X)
        Xt_hydra = self._scale_hydra.transform(Xt_hydra)

        Xt_multirocket = self._transform_multirocket.transform(X)
        Xt_multirocket = self._scale_multirocket.transform(Xt_multirocket)

        Xt = np.concatenate((Xt_hydra, Xt_multirocket), axis=1)

        return self.classifier_.predict(Xt)


def _optimal_k(n_train, k_min=6000, k_max=35000, midpoint=300, steepness=0.010):
    return int(k_min + (k_max - k_min) / (1 + np.exp(-steepness * (n_train - midpoint))))


class MultiRocketHydraSelectKBestClassifier(BaseClassifier):
    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        k=None,
        n_kernels: int = 8,
        n_groups: int = 64,
        n_jobs: int = 1,
        random_state=None,
        classifier=None,
    ):
        self.k = k
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.classifier = classifier

        super().__init__()

    def _fit(self, X, y):
        self._n_jobs = check_n_jobs(self.n_jobs)

        t0 = time.perf_counter()
        self._transform_hydra = HydraTransformer(
            n_kernels=self.n_kernels,
            n_groups=self.n_groups,
            n_jobs=self._n_jobs,
            random_state=self.random_state,
        )
        Xt_hydra = self._transform_hydra.fit_transform(X)
        self._scale_hydra = _SparseScaler()
        Xt_hydra = self._scale_hydra.fit_transform(Xt_hydra)
        self.hydra_time_ = time.perf_counter() - t0

        t0 = time.perf_counter()
        self._transform_multirocket = MultiRocket(
            n_jobs=self._n_jobs,
            random_state=self.random_state,
        )
        Xt_multirocket = self._transform_multirocket.fit_transform(X)
        self._scale_multirocket = StandardScaler()
        Xt_multirocket = self._scale_multirocket.fit_transform(Xt_multirocket)
        self.multirocket_time_ = time.perf_counter() - t0

        Xt = np.concatenate((Xt_hydra, Xt_multirocket), axis=1)

        k = self.k if self.k is not None else _optimal_k(X.shape[0])
        self.k_ = k

        clf = self.classifier if self.classifier is not None else RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.classifier_ = Pipeline([
            ("var", VarianceThreshold()),
            ("select", SelectKBest(f_classif, k=k)),
            ("clf", clf),
        ])

        t0 = time.perf_counter()
        self.classifier_.fit(Xt, y)
        self.classifier_time_ = time.perf_counter() - t0

        return self

    def _predict(self, X) -> np.ndarray:
        Xt_hydra = self._transform_hydra.transform(X)
        Xt_hydra = self._scale_hydra.transform(Xt_hydra)

        Xt_multirocket = self._transform_multirocket.transform(X)
        Xt_multirocket = self._scale_multirocket.transform(Xt_multirocket)

        Xt = np.concatenate((Xt_hydra, Xt_multirocket), axis=1)

        return self.classifier_.predict(Xt)
