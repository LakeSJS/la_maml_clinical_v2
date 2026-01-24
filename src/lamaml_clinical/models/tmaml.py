"""Temporal Model-Agnostic Meta-Learning (TMAML) module."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import higher
import torch

from lamaml_clinical.models.base import BaseModule


class TmamlModule(BaseModule):
    """
    Temporal MAML module with inner/outer loop optimization.

    Implements Temporal MAML-style meta-learning with:
    - Inner loop: One-sample-at-a-time gradient descent
    - Outer loop: Meta-optimization on future samples (optionally combined)

    Args:
        model: The underlying transformer model
        tokenizer: Tokenizer for the model
        inner_loop_learning_rate: Learning rate for inner loop adaptation
        learning_rate: Learning rate for meta-optimizer (outer loop)
        buffer_size: Size of the replay buffer
        future_step: Number of years to look ahead
        local_sample_limit: Optional limit on samples per training phase
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        inner_loop_learning_rate: float = 1e-5,
        learning_rate: float = 1e-5,
        buffer_size: int = 500,
        future_step: int = 1,
        local_sample_limit: Optional[int] = None,
    ):
        super().__init__(model, tokenizer, learning_rate, buffer_size, local_sample_limit)
        self.inner_loop_learning_rate = inner_loop_learning_rate
        self.automatic_optimization = False
        self.save_hyperparameters()

        self.meta_optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        self.future_step = future_step
        self.meta_include_current = False

    def build_meta_batch(
        self,
        batch: Dict[str, torch.Tensor],
        future_batch: Dict[str, torch.Tensor],
        include_current: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Build a meta-batch from future samples (optionally combine with current).

        Args:
            batch: Current training batch
            future_batch: Future training batch
            include_current: If True, concatenate current + future samples

        Returns:
            Meta-batch for outer-loop optimization
        """
        if not include_current:
            return future_batch
        return {
            key: torch.cat([batch[key], future_batch[key]], dim=0) for key in batch
        }

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """
        TMAML training step with inner/outer loop.

        Args:
            batch: Training batch with input_ids, attention_mask, labels
            batch_idx: Index of the batch
            dataloader_idx: Index of the dataloader

        Returns:
            Meta loss
        """
        if isinstance(batch, dict) and "current" in batch and "future" in batch:
            current_batch = batch["current"]
            future_batch = batch["future"]
        elif isinstance(batch, (list, tuple)):
            current_batch = batch[dataloader_idx]
            future_idx = dataloader_idx + self.future_step
            if future_idx >= len(batch):
                raise ValueError(
                    f"TMAML expected a future batch at index {future_idx}, "
                    f"but only {len(batch)} loaders were provided."
                )
            future_batch = batch[future_idx]
        else:
            raise ValueError(
                "TMAML expects a batch dict with keys {'current', 'future'} or "
                "a list/tuple of dataloader batches."
            )

        loss_fn = self.base_criterion()

        # Build meta-batch with future samples
        meta_batch = self.build_meta_batch(
            current_batch,
            future_batch,
            include_current=self.meta_include_current,
        )

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
                current_batch["input_ids"].size(0),
                device=current_batch["input_ids"].device,
            )

            # Inner loop: one sample at a time
            for idx in idx_order:
                x = current_batch["input_ids"][idx : idx + 1]
                m = current_batch["attention_mask"][idx : idx + 1]
                y = current_batch["labels"][idx : idx + 1]

                logits = fmodel(x, m).logits
                loss_fn = self.base_criterion()
                loss = loss_fn(logits, y)

                diffopt.step(loss, override={"lr": [self.inner_loop_learning_rate]})

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
