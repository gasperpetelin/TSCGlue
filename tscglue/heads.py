"""Final heads that combine member probabilities.

Names follow ``<method>[-T]-<loss>-<members>``: ``mean``, ``wmean`` (temperature + weights)
or ``caruana`` (greedy ensemble selection); ``-T`` = every input temperature-scaled; ``ll`` =
fit by log loss; members ``L1`` (the 4 stackers of the balanced mean), ``L0L1`` (all base
models and stackers) or ``L2`` (the heads themselves).

Every head takes member probabilities ``P`` of shape ``(n_members, n_samples, n_classes)``
and labels ``y`` as class indices.
"""

from collections import Counter

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_softmax, softmax

EPS = 1e-12


def _combine(logP, a, t):
    """softmax(a)-weighted mean of the members' softmax(logP / exp(t)), plus the parts the
    gradient needs."""
    Z = logP / np.exp(t)[:, None, None]
    Q = np.exp(log_softmax(Z, axis=2))
    w = softmax(a)
    return np.tensordot(w, Q, axes=1), Q, Z, w


class WMean:
    """Per-member temperature + softmax weights, fit by log loss with an N(0, 1) prior on the log
    temperatures and weight logits (so it shrinks to a plain mean). ``weights=False`` fits
    temperatures only, with equal weights."""

    def __init__(self, weights=True):
        self.weights = weights

    def fit(self, P, y):
        logP = np.log(np.clip(P, EPS, 1))
        M, rows = P.shape[0], np.arange(len(y))

        def loss_and_grad(theta):
            a, t = (theta[:M], theta[M:]) if self.weights else (np.zeros(M), theta)
            mix, Q, Z, w = _combine(logP, a, t)
            q = mix[rows, y]
            # the clamp has zero gradient below EPS
            inv = np.where(q > EPS, 1 / np.maximum(q, EPS), 0.0)
            Qy = Q[:, rows, y]
            g_w = -(Qy * inv).sum(axis=1)
            g_t = (w[:, None] * Qy * inv * (Z[:, rows, y] - (Q * Z).sum(axis=2))).sum(axis=1) + t
            loss = -np.log(np.maximum(q, EPS)).sum() + 0.5 * (theta ** 2).sum()
            if not self.weights:
                return loss, g_t
            return loss, np.concatenate([w * (g_w - w @ g_w) + a, g_t])

        theta0 = np.zeros(2 * M if self.weights else M)
        res = minimize(loss_and_grad, theta0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "gtol": 1e-7, "ftol": 1e-12})
        theta = res.x if np.all(np.isfinite(res.x)) else theta0
        self.a_, self.t_ = (theta[:M], theta[M:]) if self.weights else (np.zeros(M), theta)
        return self

    def predict(self, P):
        return _combine(np.log(np.clip(P, EPS, 1)), self.a_, self.t_)[0]


def _selection_loss(y, p, binary):
    """AutoGluon's log loss: binary on the positive-class column, multiclass as sklearn's."""
    if binary:
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    eps = np.finfo(np.float64).eps
    return float(-np.log(np.clip(p[np.arange(len(y)), y], eps, 1 - eps)).mean())


class Caruana:
    """Greedy ensemble selection by log loss (Caruana et al. 2004), re-implemented from
    AutoGluon's ``EnsembleSelection`` with its defaults: 25 rounds with replacement, random
    tie-breaking that prefers members already picked, rounded scores after round 1, and the
    ensemble cut to its best round. ``temper=True`` first temperature-scales every member on
    its own."""

    def __init__(self, temper=True, ensemble_size=25, seed=0):
        self.temper, self.ensemble_size, self.seed = temper, ensemble_size, seed

    def _members(self, P):
        if not self.temper:
            return P
        return np.stack([t.predict(p[None]) for t, p in zip(self.temps_, P)])

    def fit(self, P, y):
        if self.temper:
            self.temps_ = [WMean(weights=False).fit(p[None], y) for p in P]
        P = self._members(P)
        binary = P.shape[2] == 2
        preds = [p[:, 1] for p in P] if binary else list(P)
        rng = np.random.RandomState(self.seed)
        ensemble, trajectory, order, used = [], [], [], set()
        ens_pred = np.zeros(preds[0].shape)
        weighted = np.zeros(preds[0].shape)
        round_scores = False
        for i in range(self.ensemble_size):
            s = len(ensemble)
            if s:
                ens_pred *= (s - 1) / s
                ens_pred += ensemble[-1] / s
                weighted[:] = (s / (s + 1)) * ens_pred
            scores = np.zeros(len(preds))
            for j, pred in enumerate(preds):
                fant = weighted + pred / (s + 1)
                if not binary:
                    fant = fant / fant.sum(axis=1)[:, None]
                scores[j] = _selection_loss(y, fant, binary)
                if round_scores:
                    scores[j] = scores[j].round(6)
            best_all = np.flatnonzero(np.isclose(scores, np.nanmin(scores), atol=0, rtol=1e-12))
            if len(best_all) > 1 and used:
                reused = [m for m in best_all if m in used]
                best_all = reused or best_all
            best = rng.choice(best_all)
            best_score = scores[best]
            if i == 0 and np.abs(best_score) > 1e-4:
                round_scores = True
                best_score = best_score.round(6)
            ensemble.append(preds[best])
            trajectory.append(best_score)
            order.append(best)
            used.add(best)
            if len(preds) == 1:
                break
        size = trajectory.index(min(trajectory)) + 1
        self.weights_ = np.zeros(len(preds))
        for m, count in Counter(order[:size]).items():
            self.weights_[m] = count / size
        return self

    def predict(self, P):
        return np.tensordot(self.weights_, self._members(P), axes=1)


class Mean:
    """Plain average of the members."""

    def fit(self, P, y):
        return self

    def predict(self, P):
        return P.mean(axis=0)


class Blend:
    """``wmean-T-ll-L2``: temperature + weights over candidate heads, fit on their held-out
    predictions.

    ``candidates`` maps a head name to ``(make_head, member_indices)``. For every
    ``(train, val)`` split in ``splits``, each candidate is fit on ``train`` and predicts
    ``val``, and the blend is fit on those held-out predictions. At predict time each candidate
    is the mean of its fold fits (no refit on all rows).
    """

    def __init__(self, candidates):
        self.candidates = candidates

    def fit(self, P, y, splits):
        held = {c: np.zeros(P.shape[1:]) for c in self.candidates}
        self.fold_heads_ = {c: [] for c in self.candidates}
        for tr, va in splits:
            for c, (make_head, idx) in self.candidates.items():
                head = make_head().fit(P[idx][:, tr], y[tr])
                held[c][va] = head.predict(P[idx][:, va])
                self.fold_heads_[c].append(head)
        self.blend_ = WMean().fit(np.stack([held[c] for c in self.candidates]), y)
        return self

    def predict(self, P):
        heads = [np.mean([h.predict(P[idx]) for h in self.fold_heads_[c]], axis=0)
                 for c, (_, idx) in self.candidates.items()]
        return self.blend_.predict(np.stack(heads))

    @property
    def weights_(self):
        return dict(zip(self.candidates, softmax(self.blend_.a_)))


def blend_candidates(l1_idx, all_idx):
    """The 5 candidates of ``wmean-T-ll-L2``: ``mean-L1``, ``wmean-T-ll-{L1,L0L1}`` and
    ``caruana-T-ll-{L1,L0L1}``."""
    return {
        "mean-L1": (Mean, l1_idx),
        "wmean-T-ll-L1": (WMean, l1_idx),
        "wmean-T-ll-L0L1": (WMean, all_idx),
        "caruana-T-ll-L1": (Caruana, l1_idx),
        "caruana-T-ll-L0L1": (Caruana, all_idx),
    }
