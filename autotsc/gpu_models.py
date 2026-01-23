import numpy as np
import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


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
