"""Continual Model-Agnostic Meta-Learning (CMAML) module."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import higher
import torch

from lamaml_clinical.models.base import BaseModule


class CmamlModule(BaseModule):
    """
    Continual MAML module with inner/outer loop optimization.

    Implements MAML-style meta-learning with:
    - Inner loop: One-sample-at-a-time gradient descent
    - Outer loop: Meta-optimization on combined current + replay samples

    Args:
        model: The underlying transformer model
        tokenizer: Tokenizer for the model
        inner_loop_learning_rate: Learning rate for inner loop adaptation
        learning_rate: Learning rate for meta-optimizer (outer loop)
        buffer_size: Size of the replay buffer
        local_sample_limit: Optional limit on samples per training phase
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        inner_loop_learning_rate: float = 1e-5,
        learning_rate: float = 1e-5,
        buffer_size: int = 500,
        local_sample_limit: Optional[int] = None,
    ):
        super().__init__(model, tokenizer, learning_rate, buffer_size, local_sample_limit)
        self.inner_loop_learning_rate = inner_loop_learning_rate
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["model", "tokenizer"])

        self.meta_optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """
        CMAML training step with inner/outer loop.

        Args:
            batch: Training batch with input_ids, attention_mask, labels
            batch_idx: Index of the batch
            dataloader_idx: Index of the dataloader

        Returns:
            Meta loss
        """
        if isinstance(batch, list):
            batch = batch[dataloader_idx]

        labels = batch["labels"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        loss_fn = self.base_criterion()

        # Build meta-batch with replay buffer samples
        meta_batch = self.build_meta_batch(batch)

        # Get optimizer and prepare for inner loop
        opt = self.optimizers()
        base_opt = opt.optimizer
        opt.zero_grad(set_to_none=True)

        with higher.innerloop_ctx(
            self.model,
            base_opt,
            copy_initial_weights=False,
            track_higher_grads=True,
        ) as (fmodel, diffopt):
            # Randomize order for inner loop
            idx_order = torch.randperm(
                batch["input_ids"].size(0), device=batch["input_ids"].device
            )

            # Inner loop: one sample at a time
            for idx in idx_order:
                x = batch["input_ids"][idx : idx + 1]
                m = batch["attention_mask"][idx : idx + 1]
                y = batch["labels"][idx : idx + 1]

                logits = fmodel(x, m).logits
                loss_fn = self.base_criterion()
                loss = loss_fn(logits, y)

                diffopt.step(loss)

                # Update replay buffer (first epoch only)
                if self.current_epoch == 0:
                    self.replay_buffer.expose_to_sample({
                        "input_ids": x.detach(),
                        "attention_mask": m.detach(),
                        "labels": y.detach(),
                    })

            # Outer loop: compute meta loss on combined batch
            meta_logits = fmodel(
                meta_batch["input_ids"], meta_batch["attention_mask"]
            ).logits
            meta_loss = loss_fn(meta_logits, meta_batch["labels"])
            self.manual_backward(meta_loss)

        # Update meta-optimizer
        opt.step()

        self.log("meta_loss", meta_loss, prog_bar=True, on_step=True, on_epoch=True)
        return meta_loss

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure meta-optimizer."""
        return [self.meta_optimizer]
