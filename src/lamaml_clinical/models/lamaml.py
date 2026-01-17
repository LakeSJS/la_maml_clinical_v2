"""Lookahead MAML (LA-MAML) module."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import higher
import torch
import torch.nn as nn

from lamaml_clinical.models.base import BaseModule


def clip_grads(grads: List[torch.Tensor], max_norm: float = 2.0) -> List[torch.Tensor]:
    """
    Clip gradients by global norm.

    Args:
        grads: List of gradient tensors
        max_norm: Maximum allowed norm

    Returns:
        List of clipped gradient tensors
    """
    total_norm = torch.norm(torch.stack([g.norm() for g in grads]))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        grads = [g * scale for g in grads]
    return grads


class LamamlModule(BaseModule):
    """
    LA-MAML module with per-parameter learnable learning rates.

    Key innovation: Instead of a single fixed learning rate, each parameter
    has its own learnable learning rate that adapts during training.

    Two-level optimization:
    - Level 1 (Inner): Adapt model weights using learnable learning rates
    - Level 2 (Outer): Update learnable LRs and meta-parameters

    Args:
        model: The underlying transformer model
        tokenizer: Tokenizer for the model
        nu_lr: Learning rate for updating the learning rates
        alpha_0: Initial value for per-parameter learning rates
        buffer_size: Size of the replay buffer
        local_sample_limit: Optional limit on samples per training phase
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        nu_lr: float = 1e-5,
        alpha_0: float = 1e-5,
        buffer_size: int = 500,
        local_sample_limit: Optional[int] = None,
    ):
        super().__init__(
            model, tokenizer, learning_rate=alpha_0, buffer_size=buffer_size,
            local_sample_limit=local_sample_limit
        )
        self.nu_lr = nu_lr
        self.alpha_0 = alpha_0
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["model", "tokenizer"])

        # Learnable learning rates for each parameter
        self.learnable_lrs = nn.ParameterList([
            nn.Parameter(torch.full_like(p.data, self.alpha_0))
            for p in self.model.parameters()
            if p.requires_grad
        ])

        # Optimizer for learnable learning rates
        self.lr_optimizer = torch.optim.SGD(self.learnable_lrs, lr=self.nu_lr)

        # Meta-optimizer for model parameters (lr=1.0 because we scale by learnable LRs)
        self.meta_optimizer = torch.optim.SGD(
            [{"params": [p], "lr": 1.0} for p in self.model.parameters() if p.requires_grad]
        )

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> torch.Tensor:
        """
        LA-MAML training step with learnable learning rates.

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

        # Get optimizers
        meta_opt, lr_opt = self.optimizers()
        base_opt = meta_opt.optimizer
        meta_opt.zero_grad(set_to_none=True)
        lr_opt.zero_grad(set_to_none=True)

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

            # Inner loop: one sample at a time with learnable learning rates
            for idx in idx_order:
                x = batch["input_ids"][idx : idx + 1]
                m = batch["attention_mask"][idx : idx + 1]
                y = batch["labels"][idx : idx + 1]

                logits = fmodel(x, m).logits
                loss_fn = self.base_criterion()
                loss = loss_fn(logits, y)

                # Inner step with learnable learning rates and gradient clipping
                diffopt.step(
                    loss,
                    override={"lr": self.learnable_lrs},
                    create_graph=True,
                    grad_callback=clip_grads,
                )

                # Update replay buffer (first epoch only)
                if self.current_epoch == 0:
                    self.replay_buffer.expose_to_sample({
                        "input_ids": x.detach(),
                        "attention_mask": m.detach(),
                        "labels": y.detach(),
                    })

            # Compute meta loss on combined batch
            meta_logits = fmodel(
                meta_batch["input_ids"], meta_batch["attention_mask"]
            ).logits
            meta_loss = loss_fn(meta_logits, meta_batch["labels"])
            self.manual_backward(meta_loss, retain_graph=True)

        # Update learning rates
        torch.nn.utils.clip_grad_norm_(self.learnable_lrs, max_norm=2.0)
        lr_opt.step()

        # Update model parameters (outer loop)
        # Clamp learning rates to be non-negative for the meta step only
        clamped_lrs = [torch.clamp(lr, min=0.0).detach() for lr in self.learnable_lrs]
        with torch.no_grad():
            # Only iterate over trainable params to match learnable_lrs
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            for p, lr in zip(trainable_params, clamped_lrs):
                if p.grad is not None:
                    p.grad.mul_(lr.detach())

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
        meta_opt.step()

        # Zero gradients for next iteration
        lr_opt.zero_grad(set_to_none=True)
        meta_opt.zero_grad(set_to_none=True)

        self.log("meta_loss", meta_loss, prog_bar=True, on_step=True, on_epoch=True)
        return meta_loss

    def configure_optimizers(self) -> List[torch.optim.Optimizer]:
        """Configure meta-optimizer and LR optimizer."""
        return [self.meta_optimizer, self.lr_optimizer]
