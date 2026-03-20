import os
import sys
import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from chronos import BaseChronosPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from threadpoolctl import threadpool_limits
from sklearn.utils.extmath import softmax
from lightgbm import LGBMClassifier

# TODO it is duplicated here. remove in the future.
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

class Chronos2Classifier:
    # TODO This is pure AI slop and should be replaced with a proper Chronos2-based classifier.
    def __init__(self):
        MODEL_ID = "amazon/chronos-2"
        self.pipeline = BaseChronosPipeline.from_pretrained(
            MODEL_ID,
            device_map="auto",
        )

    def _pool_to_vector(self, t: torch.Tensor) -> torch.Tensor:
        # embed() returns (n_variates, num_patches+2, d_model) per series.
        # Layout: [context patches ... | REG token | masked future patch]
        # Use the [REG] token at [-2]: it attends over all context patches and
        # acts as a CLS-style sequence summary, which is best for classification.
        if t.ndim == 3:
            if t.shape[0] == 1:
                t = t[0]   # (num_patches+2, d_model)
            else:
                t = t.mean(dim=0)
        if t.ndim == 2:
            return t[-2]   # [REG] token, shape (d_model,)
        if t.ndim == 1:
            return t
        return t.reshape(-1)

    def build_long_df(self, batch: list[torch.Tensor]) -> pd.DataFrame:
        rows = []
        for item_id, ts in enumerate(batch):
            values = ts.detach().cpu().numpy().astype(np.float32)
            for t, v in enumerate(values):
                rows.append(
                    {
                        "item_id": f"series_{item_id}",
                        "timestamp": pd.Timestamp("2000-01-01") + pd.Timedelta(days=t),
                        "target": float(v),
                    }
                )
        return pd.DataFrame(rows)

    def extract_series_embedding_matrix(self, embed_output) -> torch.Tensor:
        # Chronos2Pipeline.embed currently returns:
        # (list[tensor per input], list[...aux...])
        if isinstance(embed_output, tuple) and len(embed_output) > 0 and isinstance(embed_output[0], list):
            first = embed_output[0]
            tensors = [x for x in first if isinstance(x, torch.Tensor)]
            if tensors:
                return torch.stack([self._pool_to_vector(t) for t in tensors], dim=0)

        queue = [embed_output]
        vectors = []
        while queue:
            obj = queue.pop(0)
            if isinstance(obj, torch.Tensor):
                vectors.append(self._pool_to_vector(obj))
                continue
            if isinstance(obj, np.ndarray):
                vectors.append(self._pool_to_vector(torch.from_numpy(obj)))
                continue
            if isinstance(obj, dict):
                queue.extend(obj.values())
                continue
            if isinstance(obj, (list, tuple)):
                queue.extend(obj)
                continue
            for attr in ("embeddings", "last_hidden_state", "hidden_states"):
                if hasattr(obj, attr):
                    queue.append(getattr(obj, attr))

        if vectors:
            return torch.stack(vectors, dim=0)
        raise TypeError(f"Could not locate embeddings tensor(s) in output type {type(embed_output)}")

    def run_embed_with_fallbacks(self, batch: list[torch.Tensor]):
        padded = pad_sequence(batch, batch_first=True)
        df = self.build_long_df(batch)

        attempts = [
            ("embed(context=list[Tensor])", lambda: self.pipeline.embed(context=batch)),
            ("embed(list[Tensor])", lambda: self.pipeline.embed(batch)),
            ("embed(context=Tensor[B,T])", lambda: self.pipeline.embed(context=padded)),
            ("embed(Tensor[B,T])", lambda: self.pipeline.embed(padded)),
            ("embed(df=..., target='target')", lambda: self.pipeline.embed(
                df=df, id_column="item_id", timestamp_column="timestamp", target="target"
            )),
            ("embed(df=..., target_column='target')", lambda: self.pipeline.embed(
                df=df, id_column="item_id", timestamp_column="timestamp", target_column="target"
            )),
            ("embed(df)", lambda: self.pipeline.embed(df)),
        ]

        errors = {}
        for label, fn in attempts:
            try:
                out = fn()
                # print(f"Succeeded with: {label}")
                return out, label
            except Exception as e:
                errors[label] = repr(e)

        raise RuntimeError("All embed attempts failed:\n" + "\n".join(f"- {k}: {v}" for k, v in errors.items()))

    def compute_chronos2_embeddings_local(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        embs = []
        for i in range(0, X.shape[0], batch_size):
            batch_np = X[i : i + batch_size]
            batch_ts = [torch.from_numpy(series.squeeze(0)).float() for series in batch_np]
            raw_embed_output, _ = self.run_embed_with_fallbacks(batch_ts)
            embed_matrix = self.extract_series_embedding_matrix(raw_embed_output)
            embs.append(embed_matrix.detach().cpu().float().numpy())
        return np.vstack(embs)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xt = self.compute_chronos2_embeddings_local(X)
        self.scaler_ = StandardScaler()
        Xt = self.scaler_.fit_transform(Xt)
        self.clf_ = RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 13))
        self.clf_.fit(Xt, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xt = self.compute_chronos2_embeddings_local(X)
        Xt = self.scaler_.transform(Xt)
        return self.clf_.predict(Xt)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xt = self.compute_chronos2_embeddings_local(X)
        Xt = self.scaler_.transform(Xt)
        return self.clf_.predict_proba(Xt)


# --- Embedding extractors ---

def _mantis_embed(X: np.ndarray) -> np.ndarray:
    from mantis.architecture import MantisV2
    from mantis.trainer import MantisTrainer
    network = MantisV2(device='cpu')
    network = network.from_pretrained("paris-noah/MantisV2")
    model = MantisTrainer(device='cpu', network=network)
    X_r = F.interpolate(
        torch.tensor(X, dtype=torch.float), size=512, mode='linear', align_corners=False
    ).numpy()
    return model.transform(X_r)


def _chronos2_embed(X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    all_embs = []
    for i in range(0, len(X), batch_size):
        batch = [torch.from_numpy(x.squeeze(0)).float() for x in X[i:i+batch_size]]
        embeddings, _ = pipeline.embed(batch)
        vecs = [e[0, -2, :].detach().cpu().numpy() for e in embeddings]
        all_embs.append(np.stack(vecs))
    return np.vstack(all_embs)


class Chronos2Embedding(BaseEstimator, TransformerMixin):
    """Extracts Chronos [REG] token embeddings.

    Chronos-2 (decoder-only): [REG] at [-2]
    Chronos-Bolt (encoder-decoder): [REG] at [-1]
    """
    def __init__(self, model_id="amazon/chronos-2", batch_size=256, device="cpu", include_diff=True):
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device
        self.include_diff = include_diff

    @property
    def _is_bolt(self):
        return "bolt" in self.model_id

    def fit(self, X, y=None):
        return self

    def _embed(self, X):
        pipeline = BaseChronosPipeline.from_pretrained(self.model_id, device_map=self.device)
        reg_idx = -1 if self._is_bolt else -2

        all_embs = []
        for i in range(0, len(X), self.batch_size):
            batch = [torch.from_numpy(x.squeeze(0)).float() for x in X[i:i+self.batch_size]]
            embeddings, _ = pipeline.embed(batch)

            if self._is_bolt:
                vecs = embeddings[:, reg_idx, :].detach().cpu().float().numpy()
            else:
                vecs = np.stack([e[0, reg_idx, :].detach().cpu().float().numpy() for e in embeddings])

            all_embs.append(vecs)

        return np.vstack(all_embs)

    def transform(self, X):
        emb = self._embed(X)
        if self.include_diff:
            diff_emb = self._embed(np.diff(X, axis=-1))
            return np.hstack([emb, diff_emb])
        return emb


class MantisEmbedding(BaseEstimator, TransformerMixin):
    """Extracts MantisV2 frozen embeddings (256d). Resizes input to 512 timesteps."""

    def __init__(self, device="cpu", include_diff=True):
        self.device = device
        self.include_diff = include_diff

    def fit(self, X, y=None):
        return self

    def _embed(self, X):
        from mantis.architecture import MantisV2
        from mantis.trainer import MantisTrainer

        network = MantisV2(device=self.device)
        network = network.from_pretrained("paris-noah/MantisV2")
        model = MantisTrainer(device=self.device, network=network)
        X_r = F.interpolate(
            torch.tensor(X, dtype=torch.float), size=512, mode='linear', align_corners=False
        ).numpy()
        return model.transform(X_r)

    def transform(self, X):
        emb = self._embed(X)
        if self.include_diff:
            diff_emb = self._embed(np.diff(X, axis=-1))
            return np.hstack([emb, diff_emb])
        return emb


EMBEDDING_FUNCS = {
    "mantis": _mantis_embed,
    "chronos2": _chronos2_embed,
    "mantis+chronos2": lambda X: np.hstack([_mantis_embed(X), _chronos2_embed(X)]),
}

CLASSIFIERS = {
    "ridgecv": lambda rs: RidgeClassifierCVDecisionProba(alphas=np.logspace(-3, 3, 10)),
    "rf": lambda rs: RandomForestClassifier(n_estimators=500, random_state=rs, n_jobs=-1),
    "et": lambda rs: ExtraTreesClassifier(n_estimators=500, random_state=rs, n_jobs=-1),
    "hgb": lambda rs: HistGradientBoostingClassifier(max_iter=500, random_state=rs),
    "lgbm": lambda rs: LGBMClassifier(n_estimators=500, random_state=rs, verbose=-1),
}


def _diff_series(X: np.ndarray) -> np.ndarray:
    """First difference along time axis. (n, 1, length) -> (n, 1, length-1)."""
    return np.diff(X, axis=-1)


class EmbeddingClassifier:
    def __init__(self, embedding_name, classifier_name, random_state=42, use_diff=False):
        self.embedding_name = embedding_name
        self.classifier_name = classifier_name
        self.random_state = random_state
        self.use_diff = use_diff

    def _embed(self, X):
        raw = EMBEDDING_FUNCS[self.embedding_name](X)
        if not self.use_diff:
            return raw
        diffed = EMBEDDING_FUNCS[self.embedding_name](_diff_series(X))
        return np.hstack([raw, diffed])

    def fit(self, X, y):
        Xt = self._embed(X)
        self.pipe_ = make_pipeline(StandardScaler(), CLASSIFIERS[self.classifier_name](self.random_state))
        self.pipe_.fit(Xt, y)
        return self

    def predict(self, X):
        Xt = self._embed(X)
        return self.pipe_.predict(Xt)

    def predict_proba(self, X):
        Xt = self._embed(X)
        return self.pipe_.predict_proba(Xt)


# Generate all embedding + classifier combinations as named classes
ALL_TSFM_MODELS = {}
for _emb_name in EMBEDDING_FUNCS:
    for _clf_name in CLASSIFIERS:
        _model_name = f"{_emb_name}-{_clf_name}"
        ALL_TSFM_MODELS[_model_name] = (_emb_name, _clf_name)


def download_models():
    """Pre-download all HF models to local cache (run before offline/SLURM)."""
    import time
    from huggingface_hub import snapshot_download, hf_hub_download

    repos = [
        "paris-noah/MantisV2",
        "amazon/chronos-2",
        "amazon/chronos-bolt-tiny",
        "amazon/chronos-bolt-mini",
        "amazon/chronos-bolt-small",
        "amazon/chronos-bolt-base",
    ]

    for i, repo in enumerate(repos, 1):
        print(f"[{i}/{len(repos) + 1}] Downloading {repo} ...", flush=True)
        t0 = time.time()
        snapshot_download(repo)
        print(f"[{i}/{len(repos) + 1}] Cached {repo} in {time.time() - t0:.1f}s", flush=True)

    print(f"[{len(repos) + 1}/{len(repos) + 1}] Downloading jingang/TabICL ...", flush=True)
    t0 = time.time()
    ckpt = hf_hub_download(repo_id="jingang/TabICL", filename="tabicl-classifier-v2-20260212.ckpt")
    print(f"[{len(repos) + 1}/{len(repos) + 1}] Cached jingang/TabICL -> {ckpt} in {time.time() - t0:.1f}s", flush=True)

    print("All models cached.", flush=True)


class TabICLTimeSeriesClassifier(BaseEstimator):
    """TabICLv2 on tabularized time series (aeon 3D -> 2D via Tabularizer)."""

    def __init__(self, random_state=42, device="cpu", n_jobs=1, kv_cache=True, batch_size=None, include_diff=False):
        self.random_state = random_state
        self.device = device
        self.n_jobs = n_jobs
        self.kv_cache = kv_cache
        self.batch_size = batch_size
        self.include_diff = include_diff

    def _prepare_X(self, X):
        if not self.include_diff:
            return X
        X_diff = np.diff(X, axis=-1)
        return np.concatenate([X, X_diff], axis=-1)

    def _make_pipe(self):
        from aeon.transformations.collection import Tabularizer
        from tabicl import TabICLClassifier

        return make_pipeline(
            Tabularizer(),
            TabICLClassifier(
                device=self.device,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
                kv_cache=self.kv_cache,
                batch_size=self.batch_size,
            ),
        )

    def fit(self, X, y):
        self.pipe_ = self._make_pipe()
        self.pipe_.fit(self._prepare_X(X), y)
        return self

    def predict(self, X):
        return self.pipe_.predict(self._prepare_X(X))

    def predict_proba(self, X):
        return self.pipe_.predict_proba(self._prepare_X(X))


def make_tsfm_model(model_name: str, random_state: int = 42, use_diff: bool = False) -> EmbeddingClassifier:
    if model_name not in ALL_TSFM_MODELS:
        raise ValueError(f"Unknown tsfm model: {model_name}. Available: {list(ALL_TSFM_MODELS.keys())}")
    emb_name, clf_name = ALL_TSFM_MODELS[model_name]
    return EmbeddingClassifier(emb_name, clf_name, random_state=random_state, use_diff=use_diff)