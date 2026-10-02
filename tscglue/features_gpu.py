"""Device-aware feature transformers.

``get_feature_transformer("hydra", device=...)`` returns ``HydraTransformerDevice``
instead of aeon's transformer whenever a non-cpu device is asked for, which is what
``TSCGlueEnhancedV4`` uses to move hydra onto the GPU.
"""

import warnings

import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer

from tscglue.utils import require_torch as _require_torch

__all__ = ["HydraTransformerDevice"]

_KERNEL_LENGTH = 9


class HydraTransformerDevice(BaseCollectionTransformer):
    """Hydra, with the device and precision as arguments.

    A reimplementation of aeon's ``HydraTransformer`` rather than a subclass of it.
    Upstream pins every allocation to CPU float32 -- it builds its input with
    ``torch.tensor(X)`` and its count buffers with a bare ``torch.zeros`` -- so steering
    it from a subclass means overriding the ambient device and default dtype around
    somebody else's ``forward``. Here they are just arguments.

    Kernels are always drawn on the CPU in float32 from a seeded local generator, so a
    given ``random_state`` gives the same kernels on every device and at every precision,
    and the outputs stay directly comparable. For an integer ``random_state`` those
    kernels are identical to upstream's.

    The transform runs in batches of ``batch_size`` cases, which upstream has the
    machinery for but never uses. Each batch is moved to the device on its own, so peak
    memory is set by the batch rather than by the size of the collection.

    Parameters
    ----------
    n_kernels : int, default=8
        Number of kernels per group.
    n_groups : int, default=64
        Number of groups per dilation.
    max_num_channels : int, default=8
        Maximum number of channels to use for each dilation.
    random_state : int or None, default=None
        Seed for the kernel draw.
    output_type : str, default="numpy"
        ``"numpy"``, ``"tensor"`` or ``"dataframe"``. Upstream defaults to ``"tensor"``.
    device : str, default="cpu"
        Torch device the transform runs on, e.g. ``"cpu"`` or ``"cuda"``.
    dtype : str, default="float32"
        Precision of the transform. ``"float64"`` is a measurement tool rather than a
        better setting -- it quantifies what float32 costs, at double the memory and
        1/32 the throughput on consumer cards.
    batch_size : int, default=256
        Cases per forward pass. Peak memory scales with ``batch_size x n_timepoints``,
        so lower it for long series on a small card.

    Notes
    -----
    Original code: https://github.com/angus924/hydra

    References
    ----------
    .. [1] Dempster, A., Schmidt, D.F. and Webb, G.I., 2023. Hydra: Competing
        convolutional kernels for fast and accurate time series classification.
        Data Mining and Knowledge Discovery, pp.1-27.

    Examples
    --------
    >>> from tscglue.features_gpu import HydraTransformerDevice
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, _ = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=64)
    >>> HydraTransformerDevice(random_state=0).fit_transform(X).shape
    (10, 3072)
    """

    _tags = {
        "capability:multivariate": True,
        "output_data_type": "Tabular",
        "algorithm_type": "convolution",
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        n_kernels=8,
        n_groups=64,
        max_num_channels=8,
        random_state=None,
        output_type="numpy",
        device="cpu",
        dtype="float32",
        batch_size=256,
    ):
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.max_num_channels = max_num_channels
        self.random_state = random_state
        self.output_type = output_type
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size

        super().__init__()

    def _fit(self, X, y=None):
        torch = _require_torch()
        n_channels, n_timepoints = X.shape[1], X.shape[2]

        generator = None
        if self.random_state is not None:
            generator = torch.Generator().manual_seed(int(self.random_state))

        max_exponent = np.log2((n_timepoints - 1) / (_KERNEL_LENGTH - 1))
        self.dilations_ = 2 ** torch.arange(int(max_exponent) + 1)
        self.paddings_ = torch.div(
            (_KERNEL_LENGTH - 1) * self.dilations_, 2, rounding_mode="floor"
        ).int()

        self.divisor_ = min(2, self.n_groups)
        self.h_ = self.n_groups // self.divisor_

        W = torch.randn(
            len(self.dilations_),
            self.divisor_,
            self.n_kernels * self.h_,
            1,
            _KERNEL_LENGTH,
            generator=generator,
        )
        W = W - W.mean(-1, keepdims=True)
        self.W_ = W / W.abs().sum(-1, keepdims=True)

        num_channels_per = int(np.clip(n_channels // 2, 2, self.max_num_channels))
        self.idx_ = [
            torch.randint(
                0, n_channels, (self.divisor_, self.h_, num_channels_per), generator=generator
            )
            for _ in range(len(self.dilations_))
        ]
        return self

    def _forward(self, X, W, idx):
        """Counts for one batch. ``X``, ``W`` and ``idx`` are already on the device."""
        torch = _require_torch()
        import torch.nn.functional as F

        n_cases, n_channels = X.shape[0], X.shape[1]
        diff_X = torch.diff(X) if self.divisor_ > 1 else None

        Z = []
        for dilation_index in range(len(self.dilations_)):
            dilation = int(self.dilations_[dilation_index])
            padding = int(self.paddings_[dilation_index])

            for diff_index in range(self.divisor_):
                source = X if diff_index == 0 else diff_X
                if n_channels > 1:
                    channels = idx[dilation_index][diff_index]
                    _Z = F.conv1d(
                        source[:, channels].sum(2),
                        W[dilation_index][diff_index],
                        dilation=dilation,
                        padding=padding,
                        groups=self.h_,
                    )
                else:
                    _Z = F.conv1d(
                        source,
                        W[dilation_index, diff_index],
                        dilation=dilation,
                        padding=padding,
                    )
                _Z = _Z.view(n_cases, self.h_, self.n_kernels, -1)

                max_values, max_indices = _Z.max(2)
                min_values, min_indices = _Z.min(2)

                counts = torch.zeros(
                    2, n_cases, self.h_, self.n_kernels, device=X.device, dtype=X.dtype
                )
                counts[0].scatter_add_(-1, max_indices, max_values)
                counts[1].scatter_add_(-1, min_indices, torch.ones_like(min_values))
                Z.append(counts[0])
                Z.append(counts[1])

        return torch.cat(Z, 1).view(n_cases, -1)

    def _resolve_device(self):
        """``self.device``, degraded to cpu when cuda is asked for but unavailable.

        A fitted transformer is pickled with its device and reloaded at predict time
        without one, so a model fitted on a GPU box would otherwise be unusable on a
        cpu-only one. ``self.device`` itself is left alone -- it is a constructor
        parameter, and sklearn clone semantics say it stays as passed.
        """
        torch = _require_torch()
        device = torch.device(self.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn(
                f"device={self.device!r} was requested but no CUDA device is available; "
                "running on cpu instead.",
                RuntimeWarning,
                stacklevel=3,
            )
            return torch.device("cpu")
        return device

    def _transform(self, X, y=None):
        torch = _require_torch()
        device, dtype = self._resolve_device(), getattr(torch, self.dtype)

        W = self.W_.to(device=device, dtype=dtype)
        idx = [i.to(device) for i in self.idx_]

        # Batches are sliced, cast and moved one at a time, and written straight into a
        # preallocated output: nothing the size of the whole collection is ever built,
        # on the host or the device.
        X = np.asarray(X)
        n_features = 2 * len(self.dilations_) * self.divisor_ * self.h_ * self.n_kernels
        Xt = torch.empty((len(X), n_features), dtype=dtype)

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                chunk = X[start : start + self.batch_size]
                if not chunk.flags.writeable:
                    # np.load(mmap_mode="r") hands back a read-only array, which
                    # torch.as_tensor only accepts with a warning.
                    chunk = chunk.copy()
                # Cast on the host before moving, so only ``dtype`` crosses the bus.
                batch = torch.as_tensor(chunk).to(dtype).to(device)
                Xt[start : start + len(batch)].copy_(self._forward(batch, W, idx))

        if self.output_type == "tensor":
            return Xt
        Xt = Xt.numpy()
        if self.output_type == "dataframe":
            import pandas as pd

            return pd.DataFrame(Xt)
        return Xt
