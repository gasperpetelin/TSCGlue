import os
import sys
import inspect
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from chronos import BaseChronosPipeline
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import RidgeClassifierCV
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from threadpoolctl import threadpool_limits
from sklearn.utils.extmath import softmax

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