"""Base module providing common functionality for all models."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC

from lamaml_clinical.models.replay_buffer import ReplayBuffer


class BaseModule(pl.LightningModule):
    """
    Base module providing common functionality for all model variants.

    Handles validation, testing, replay buffer management, and checkpointing.

    Args:
        model: The underlying transformer model
        tokenizer: Tokenizer for the model
        learning_rate: Learning rate for optimization
        buffer_size: Size of the replay buffer
        local_sample_limit: Optional limit on samples per training phase
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        learning_rate: float = 1e-5,
        buffer_size: int = 500,
        local_sample_limit: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.base_criterion = nn.CrossEntropyLoss
        self.val_auc = BinaryAUROC()
        self.test_auc = BinaryAUROC()
        self.replay_buffer = ReplayBuffer(buffer_size, local_sample_limit)
        self.save_hyperparameters(ignore=["model", "tokenizer"])

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save replay buffer state to checkpoint."""
        checkpoint["replay_buffer_size"] = self.replay_buffer.buffer_size
        checkpoint["replay_buffer_data"] = self.replay_buffer.data
        checkpoint["replay_buffer_local_samples_seen"] = self.replay_buffer.local_samples_seen
        checkpoint["replay_buffer_global_samples_seen"] = self.replay_buffer.global_samples_seen
        checkpoint["replay_buffer_local_sample_limit"] = self.replay_buffer.local_sample_limit

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load replay buffer state from checkpoint."""
        if "replay_buffer_size" in checkpoint:
            self.replay_buffer.buffer_size = checkpoint["replay_buffer_size"]
        if "replay_buffer_data" in checkpoint:
            self.replay_buffer.data = checkpoint["replay_buffer_data"]
        if "replay_buffer_local_samples_seen" in checkpoint:
            self.replay_buffer.local_samples_seen = checkpoint["replay_buffer_local_samples_seen"]
        if "replay_buffer_local_sample_limit" in checkpoint:
            self.replay_buffer.local_sample_limit = checkpoint["replay_buffer_local_sample_limit"]
        if "replay_buffer_global_samples_seen" in checkpoint:
            self.replay_buffer.global_samples_seen = checkpoint["replay_buffer_global_samples_seen"]

    def build_meta_batch(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Build a meta-batch combining current batch with replay buffer samples.

        Args:
            batch: Current training batch

        Returns:
            Combined batch with current samples and replay buffer samples
        """
        k = len(batch["input_ids"])
        buffer_samples = self.replay_buffer.get_samples(k, device=batch["input_ids"].device)
        if buffer_samples:
            meta_batch = {
                key: torch.cat([batch[key], buffer_samples[key]], dim=0) for key in batch
            }
        else:
            meta_batch = batch
        return meta_batch

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Any:
        """Forward pass through the model."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def validation_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """Validation step computing loss and AUC."""
        # Reset metric when dataloader changes to compute per-year AUC
        if not hasattr(self, "_val_dl_idx") or self._val_dl_idx != dataloader_idx:
            self.val_auc.reset()
            self._val_dl_idx = dataloader_idx

        if isinstance(batch, list):
            batch = batch[dataloader_idx]

        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        loss_fn = self.base_criterion()
        outputs = self(input_ids, attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)[:, 1]
        loss = loss_fn(logits, labels)

        self.val_auc.update(probs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log(
            "val_auc",
            self.val_auc,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def test_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Test step computing loss and AUC."""
        # Reset metric when dataloader changes to compute per-year AUC
        if not hasattr(self, "_test_dl_idx") or self._test_dl_idx != dataloader_idx:
            self.test_auc.reset()
            self._test_dl_idx = dataloader_idx

        if isinstance(batch, list):
            batch = batch[dataloader_idx]

        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        loss_fn = self.base_criterion()
        outputs = self(input_ids, attention_mask)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)[:, 1]
        loss = loss_fn(logits, labels)

        self.test_auc.update(probs, labels)
        self.log("test_loss", loss, prog_bar=True)
        self.log(
            "test_auc",
            self.test_auc,
            on_step=False,
            on_epoch=True,
        )
