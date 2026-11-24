"""Tabular classifiers used as the head of the time-series pipelines."""

# TODO import tabular models into this

from time import perf_counter

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import RidgeClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import softmax
from threadpoolctl import threadpool_limits


class RidgeClassifierCVDecisionProba(RidgeClassifierCV):
    """RidgeClassifierCV with softmax probabilities and ``n_jobs`` as a BLAS thread budget."""

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        scoring=None,
        cv=None,
        class_weight=None,
        store_cv_results=False,
        n_jobs=1,
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            scoring=scoring,
            cv=cv,
            class_weight=class_weight,
            store_cv_results=store_cv_results,
        )
        self.n_jobs = n_jobs

    def fit(self, X, y):
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            return super().fit(X, y)

    def predict_proba(self, X):
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            scores = self.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        return softmax(scores)


class RidgeClassifierCVIndicator(RidgeClassifierCV):
    """RidgeClassifierCV whose probabilities are a one-hot indicator of the predicted class."""

    def __init__(
        self,
        alphas=(0.1, 1.0, 10.0),
        *,
        fit_intercept=True,
        scoring=None,
        cv=None,
        class_weight=None,
        store_cv_results=False,
        n_jobs=1,
    ):
        super().__init__(
            alphas=alphas,
            fit_intercept=fit_intercept,
            scoring=scoring,
            cv=cv,
            class_weight=class_weight,
            store_cv_results=store_cv_results,
        )
        self.n_jobs = n_jobs

    def predict_proba(self, X):
        dists = np.zeros((X.shape[0], len(self.classes_)))
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            preds = self.predict(X)
        for i in range(0, X.shape[0]):
            dists[i, np.where(self.classes_ == preds[i])] = 1
        return dists

    def fit(self, X, y):
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            return super().fit(X, y)


class AutoSelectKBestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        classifier=None,
        k=None,
        k_min=6000,
        k_max=35000,
        midpoint=300,
        steepness=0.010,
        score_func=f_classif,
        variance_filter=True,
        n_jobs=1,
    ):
        self.classifier = classifier
        self.k = k
        self.k_min = k_min
        self.k_max = k_max
        self.midpoint = midpoint
        self.steepness = steepness
        self.score_func = score_func
        self.variance_filter = variance_filter
        self.n_jobs = n_jobs

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
        k = min(k, X.shape[1])

        if self.classifier is None:
            clf = RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10), n_jobs=self.n_jobs)
        else:
            # An explicit classifier carries its own thread budget; n_jobs is not forced on it.
            clf = clone(self.classifier)

        var = VarianceThreshold() if self.variance_filter else None
        select = SelectKBest(score_func=self.score_func, k=k)

        # Suppress f_classif "Features are constant" warnings: after CV fold splitting
        # and scaling, some features have near-zero variance that passes VarianceThreshold
        # but f_classif still treats as constant. These features are harmless (never selected).
        import warnings

        # Steps are fitted one at a time rather than through Pipeline.fit purely so each
        # can be timed; the order and the inputs are what Pipeline.fit would do. The
        # already-fitted steps are then assembled into the Pipeline predict/predict_proba
        # use, so nothing downstream changes.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # score_func (e.g. chi2's Y.T @ X) runs BLAS outside the classifier's own limit
            with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
                t0 = perf_counter()
                Xt = var.fit_transform(X) if var is not None else X
                t_var = perf_counter() - t0

                t0 = perf_counter()
                Xt = select.fit_transform(Xt, y)
                t_select = perf_counter() - t0

            t0 = perf_counter()
            clf.fit(Xt, y)
            t_clf = perf_counter() - t0

        steps = [("var", var)] if var is not None else []
        self.classifier_ = Pipeline([*steps, ("select", select), ("clf", clf)])
        self.fit_times_ = {
            "variance_s": t_var,
            "select_s": t_select,
            "clf_s": t_clf,
            "total_s": t_var + t_select + t_clf,
        }

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


class RareClassSafeLogisticCV(BaseEstimator, ClassifierMixin):
    """Logistic regression (CV over C) that survives singleton classes; ``n_jobs`` is a BLAS
    thread budget."""

    def __init__(
        self, Cs=10, fixed_C=1.0, solver="lbfgs", max_iter=1000, class_weight=None, n_jobs=1
    ):
        self.Cs = Cs
        self.fixed_C = fixed_C
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_jobs = n_jobs

    def fit(self, X, y):
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            return self._fit(X, y)

    def _fit(self, X, y):
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

        min_count = int(np.unique(y, return_counts=True)[1].min())
        if min_count < 2:
            # A singleton class would make the internal CV crash; skip it.
            self.estimator_ = LogisticRegression(
                C=self.fixed_C,
                solver=self.solver,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
            )
        else:
            # Identical to the original stacker: default cv=5, multinomial.
            self.estimator_ = LogisticRegressionCV(
                Cs=self.Cs,
                solver=self.solver,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                l1_ratios=(0.0,),
                use_legacy_attributes=False,
            )
        self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            return self.estimator_.predict(X)

    def predict_proba(self, X):
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            return self.estimator_.predict_proba(X)


class ThreadLimitedClassifier(BaseEstimator, ClassifierMixin):
    """Fit and predict ``estimator`` (default ``MLPClassifier()``) with ``n_jobs`` as a BLAS
    thread budget, for classifiers that have no ``n_jobs`` of their own."""

    def __init__(self, estimator=None, n_jobs=1):
        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator) if self.estimator is not None else MLPClassifier()
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            self.estimator_.fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            return self.estimator_.predict(X)

    def predict_proba(self, X):
        with threadpool_limits(limits=None if self.n_jobs == -1 else self.n_jobs):
            return self.estimator_.predict_proba(X)


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


class SparseScaler:
    """Sparse Scaler for hydra transform (NumPy version)."""

    def __init__(self, mask=True, exponent=4):
        self.mask = mask
        self.exponent = exponent

    def _run(self, X, fit, scale):
        Xt = np.clip(X, 0, None)
        np.sqrt(Xt, out=Xt)

        if fit:
            epsilon = (Xt == 0).mean(axis=0) ** self.exponent + 1e-8
            self.mu = Xt.mean(axis=0)
            self.sigma = (Xt.std(axis=0) + epsilon).astype(Xt.dtype)
        if not scale:
            return None

        mask = (Xt != 0) if self.mask else None  # must precede the subtraction
        Xt -= self.mu
        if mask is not None:
            Xt *= mask
        Xt /= self.sigma
        return Xt

    def fit(self, X, y=None):
        self._run(X, fit=True, scale=False)
        return self

    def transform(self, X, y=None):
        return self._run(X, fit=False, scale=True)

    def fit_transform(self, X, y=None):
        return self._run(X, fit=True, scale=True)


class NoScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
