# model.py

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy  # NEW


class MNISTClassifier(pl.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        # Avoid trying to save the entire OmegaConf object; keep only simple hparams
        if config is not None:
            self.save_hyperparameters({"LR": float(config.LR)})
        else:
            self.save_hyperparameters({"LR": 1e-3})
        self.config = config

        self.model = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10)
        )

        # Metrics (robust epoch aggregation)
        self.train_acc = Accuracy(num_classes=10)
        self.val_acc = Accuracy(num_classes=10)
        self.test_acc = Accuracy(num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.val_acc(logits, y)
        # Do NOT gate by epoch here; Trainer already controls val frequency.
        # Log on_epoch so ModelCheckpoint sees the metric.
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = self.test_acc(logits, y)
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "test_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

    def configure_optimizers(self):
        lr = float(self.config.LR) if self.config is not None else 1e-3
        return torch.optim.Adam(self.parameters(), lr=lr)
