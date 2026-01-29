import time

import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from aeon.classification import BaseClassifier
from aeon.classification.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import MultiRocket
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from aeon.utils.validation import check_n_jobs


class _LightningLinear(pl.LightningModule):
    def __init__(self, n_features, n_classes, lr, alpha, optimizer="adamw",
                 scheduler="onecycle", steps_per_epoch=None, max_epochs=None):
        super().__init__()
        self.model = nn.Linear(n_features, n_classes)
        self.lr = lr
        self.alpha = alpha
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.steps_per_epoch = steps_per_epoch
        self.max_epochs = max_epochs
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        self.log_dict({"train_loss": loss, "train_acc": acc}, prog_bar=False)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        acc = (logits.argmax(1) == y).float().mean()
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.alpha,
            )
        elif self.optimizer == "adam":
            opt = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.alpha,
            )
        else:
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.alpha,
            )

        if self.scheduler == "onecycle":
            sched = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=self.lr,
                epochs=self.max_epochs,
                steps_per_epoch=self.steps_per_epoch,
                pct_start=0.3,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

        return opt


class LightningSGDClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        lr=1e-2,
        alpha=1e-4,
        batch_size=2048,
        max_epochs=10,
        accelerator="auto",
        devices=1,
        verbose=1,
        optimizer="adamw",
        scheduler="onecycle",
    ):
        self.lr = lr
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.verbose = verbose
        self.optimizer = optimizer
        self.scheduler = scheduler

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X)
        y = np.asarray(y)

        self.le_ = LabelEncoder()
        y = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        self.n_features_in_ = X.shape[1]

        train_ds = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).long(),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
        )

        if X_val is not None:
            y_val = self.le_.transform(y_val)
            val_ds = TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).long(),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=8,
                persistent_workers=True,
                pin_memory=True,
                prefetch_factor=4,
            )
        else:
            val_loader = None

        self.model_ = _LightningLinear(
            n_features=X.shape[1],
            n_classes=len(self.classes_),
            lr=self.lr,
            alpha=self.alpha,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            steps_per_epoch=len(train_loader),
            max_epochs=self.max_epochs,
        )

        self.trainer_ = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=bool(self.verbose),
        )

        self.trainer_.fit(self.model_, train_loader, val_loader)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        X = torch.from_numpy(np.asarray(X)).float()

        loader = DataLoader(X, batch_size=self.batch_size)
        self.model_.eval()

        probs = []
        with torch.no_grad():
            for xb in loader:
                logits = self.model_(xb)
                probs.append(torch.softmax(logits, 1).cpu().numpy())

        return np.vstack(probs)

    def predict(self, X):
        p = self.predict_proba(X)
        return self.classes_[p.argmax(1)]

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


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
        print(Xt.shape)

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
