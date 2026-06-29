import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from tscglue.utils import require_torch as _require_torch


class Chronos2Embedding(BaseEstimator, TransformerMixin):
    """Extracts Chronos [REG] token embeddings.

    Chronos-2 (decoder-only): [REG] at [-2]
    Chronos-Bolt (encoder-decoder): [REG] at [-1]
    """

    def __init__(
        self,
        model_id="amazon/chronos-2",
        batch_size=256,
        device="cpu",
        include_diff=True,
        verbose=False,
    ):
        self.model_id = model_id
        self.batch_size = batch_size
        self.device = device
        self.include_diff = include_diff
        self.verbose = verbose
        self._pipeline = None

    @property
    def _is_bolt(self):
        return "bolt" in self.model_id

    def _get_pipeline(self):
        if self._pipeline is None:
            from chronos import BaseChronosPipeline

            if self.verbose:
                print(f"[Chronos2Embedding] loading {self.model_id} ...", flush=True)
            self._pipeline = BaseChronosPipeline.from_pretrained(
                self.model_id, device_map=self.device
            )
            if self.verbose:
                print("[Chronos2Embedding] model loaded", flush=True)
        return self._pipeline

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_pipeline"] = None
        return state

    def fit(self, X, y=None):
        return self

    def _embed_channel(self, X):
        torch = _require_torch()
        pipeline = self._get_pipeline()
        n_batches = (len(X) + self.batch_size - 1) // self.batch_size
        with torch.inference_mode():
            reg_idx = -1 if self._is_bolt else -2
            all_embs = []
            for batch_idx, i in enumerate(range(0, len(X), self.batch_size)):
                batch = [torch.from_numpy(x.squeeze(0)).float() for x in X[i : i + self.batch_size]]
                embeddings, _ = pipeline.embed(batch)
                if self._is_bolt:
                    vecs = embeddings[:, reg_idx, :].detach().cpu().float().numpy()
                else:
                    vecs = np.stack(
                        [e[0, reg_idx, :].detach().cpu().float().numpy() for e in embeddings]
                    )
                all_embs.append(vecs)
                if self.verbose:
                    print(f"[Chronos2Embedding] batch {batch_idx + 1}/{n_batches}", flush=True)
        return np.vstack(all_embs)

    def _embed(self, X):
        if X.shape[1] == 1:
            return self._embed_channel(X)
        return np.hstack([self._embed_channel(X[:, i : i + 1, :]) for i in range(X.shape[1])])

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
        self._model = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_model"] = None
        return state

    def fit(self, X, y=None):
        return self

    def _get_model(self):
        if self._model is None:
            from mantis.architecture import MantisV2
            from mantis.trainer import MantisTrainer

            network = MantisV2(device=self.device).from_pretrained("paris-noah/MantisV2")
            self._model = MantisTrainer(device=self.device, network=network)
        return self._model

    def _embed_channel(self, X):
        torch = _require_torch()
        import torch.nn.functional as F

        X_r = F.interpolate(
            torch.tensor(X, dtype=torch.float), size=512, mode="linear", align_corners=False
        ).numpy()
        return self._get_model().transform(X_r)

    def _embed(self, X):
        if X.shape[1] == 1:
            return self._embed_channel(X)
        return np.hstack([self._embed_channel(X[:, i : i + 1, :]) for i in range(X.shape[1])])

    def transform(self, X):
        emb = self._embed(X)
        if self.include_diff:
            diff_emb = self._embed(np.diff(X, axis=-1))
            return np.hstack([emb, diff_emb])
        return emb


def download_models():
    """Pre-download all HF models to local cache (run before offline/SLURM)."""
    import time

    from huggingface_hub import snapshot_download

    repos = [
        "paris-noah/MantisV2",
        "amazon/chronos-2",
        "amazon/chronos-bolt-tiny",
        "amazon/chronos-bolt-mini",
        "amazon/chronos-bolt-small",
        "amazon/chronos-bolt-base",
    ]

    for i, repo in enumerate(repos, 1):
        print(f"[{i}/{len(repos)}] Downloading {repo} ...", flush=True)
        t0 = time.time()
        snapshot_download(repo)
        print(f"[{i}/{len(repos)}] Cached {repo} in {time.time() - t0:.1f}s", flush=True)

    print("All models cached.", flush=True)
