import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from tscglue.utils import require_torch as _require_torch


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
        self._pipeline = None

    @property
    def _is_bolt(self):
        return "bolt" in self.model_id

    def _get_pipeline(self):
        if self._pipeline is None:
            from chronos import BaseChronosPipeline
            self._pipeline = BaseChronosPipeline.from_pretrained(self.model_id, device_map=self.device)
        return self._pipeline

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_pipeline'] = None
        return state

    def fit(self, X, y=None):
        return self

    def _embed(self, X):
        torch = _require_torch()
        pipeline = self._get_pipeline()
        with torch.inference_mode():
            reg_idx = -1 if self._is_bolt else -2
            all_embs = []
            for i in range(0, len(X), self.batch_size):
                if X.shape[1] != 1:
                    raise ValueError(f"Chronos2Embedding only supports univariate series, got {X.shape[1]} channels.")
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
        torch = _require_torch()
        import torch.nn.functional as F
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


