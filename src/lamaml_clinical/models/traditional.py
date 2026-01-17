"""Traditional fine-tuning module."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from lamaml_clinical.models.base import BaseModule


class TraditionalModule(BaseModule):
    """
    Traditional fine-tuning module with optional replay buffer population.

    This module performs standard supervised learning with AdamW optimizer.
    It populates the replay buffer during training (first epoch only) but
    does not use it for training - the buffer is maintained for transfer
    to meta-learning modules.

    Args:
        model: The underlying transformer model
        tokenizer: Tokenizer for the model
        learning_rate: Learning rate for AdamW optimizer
        buffer_size: Size of the replay buffer
        local_sample_limit: Optional limit on samples per training phase
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        learning_rate: float = 1e-5,
        buffer_size: int = 500,
        local_sample_limit: Optional[int] = None,
    ):
        super().__init__(model, tokenizer, learning_rate, buffer_size, local_sample_limit)
        self.save_hyperparameters(ignore=["model", "tokenizer"])

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """
        Training step with class-weighted loss.

        Args:
            batch: Training batch with input_ids, attention_mask, labels
            batch_idx: Index of the batch
            dataloader_idx: Index of the dataloader

        Returns:
            Training loss
        """
        if isinstance(batch, list):
            batch = batch[dataloader_idx]

        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # Apply class weights to the loss function
        class_weights = self.trainer.datamodule.train_loader_class_weights[dataloader_idx]
        loss_fn = self.base_criterion(weight=class_weights.to(self.device))

        # Forward pass and loss computation
        outputs = self(input_ids, attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        # Update replay buffer (first epoch only)
        if self.current_epoch == 0:
            for idx in range(len(labels)):
                self.replay_buffer.expose_to_sample({
                    "input_ids": input_ids[idx : idx + 1],
                    "attention_mask": attention_mask[idx : idx + 1],
                    "labels": labels[idx : idx + 1],
                })

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure AdamW optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
