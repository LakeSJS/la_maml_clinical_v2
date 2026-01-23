"""Unified experiment runner for LA-MAML clinical experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import CombinedLoader
from transformers import BertForSequenceClassification, BertTokenizer

from lamaml_clinical.core.config import ExperimentConfig, load_config
from lamaml_clinical.core.paths import ensure_dir
from lamaml_clinical.data.datamodules import TemporalDataModule
from lamaml_clinical.models import (
    BaseModule,
    CmamlModule,
    LamamlModule,
    TmamlModule,
    TraditionalModule,
)

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


MODEL_REGISTRY: Dict[str, Type[BaseModule]] = {
    "traditional": TraditionalModule,
    "cmaml": CmamlModule,
    "lamaml": LamamlModule,
    "tmaml": TmamlModule,
}


@dataclass
class ExperimentRunner:
    """
    Unified experiment runner that eliminates code duplication.

    Handles both sequential and non-sequential training modes,
    model initialization, logging, and result saving.
    """

    config: ExperimentConfig

    def __post_init__(self) -> None:
        self._setup_environment()
        self._set_seed()

    def _setup_environment(self) -> None:
        """Configure CUDA and PyTorch settings."""
        os.environ["PYTORCH_SDP_ATTENTION"] = "0"
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        torch.set_float32_matmul_precision("high")

    def _set_seed(self) -> None:
        """Set random seed for reproducibility."""
        pl.seed_everything(self.config.seed, workers=True)

    def _load_tokenizer(self) -> BertTokenizer:
        """Load tokenizer from config path."""
        return BertTokenizer.from_pretrained(self.config.paths.tokenizer_path)

    def _load_base_model(self) -> BertForSequenceClassification:
        """Load base model from config path, optionally wrapping with PEFT."""
        model = BertForSequenceClassification.from_pretrained(
            self.config.paths.base_model_id, num_labels=2
        )

        # Apply PEFT if configured
        if self.config.model.use_peft:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "PEFT is not installed. Install with: pip install peft"
                )

            model = self._apply_peft(model)
            print(f"Applied PEFT ({self.config.model.peft_method}) to base model")
            self._print_trainable_parameters(model)

        return model

    def _apply_peft(self, model: BertForSequenceClassification) -> BertForSequenceClassification:
        """
        Apply PEFT adapter to the model.

        Args:
            model: Base transformer model

        Returns:
            PEFT-wrapped model
        """
        if self.config.model.peft_method.lower() == "lora":
            # Auto-detect target modules if not specified
            target_modules = self.config.model.lora_target_modules
            if target_modules is None:
                # Default LoRA targets for BERT: query and value projection matrices
                target_modules = ["query", "value"]

            peft_config = LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                lora_dropout=self.config.model.lora_dropout,
                target_modules=target_modules,
                bias=self.config.model.lora_bias,
                inference_mode=False,
            )

            model = get_peft_model(model, peft_config)
        else:
            raise ValueError(
                f"Unsupported PEFT method: {self.config.model.peft_method}. "
                f"Currently only 'lora' is supported."
            )

        return model

    def _print_trainable_parameters(self, model: torch.nn.Module) -> None:
        """Print number of trainable parameters vs total parameters."""
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        trainable_percent = 100 * trainable_params / all_params

        print(f"Trainable params: {trainable_params:,} || "
              f"All params: {all_params:,} || "
              f"Trainable: {trainable_percent:.2f}%")

    def _create_model(
        self,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ) -> BaseModule:
        """
        Factory method to create the appropriate model module.

        Args:
            model: Optional pre-loaded model (for initialization from checkpoint)
            tokenizer: Optional tokenizer

        Returns:
            Configured model module
        """
        if tokenizer is None:
            tokenizer = self._load_tokenizer()
        if model is None:
            model = self._load_base_model()
        elif self.config.model.use_peft:
            # Apply PEFT even when model is provided (e.g., from checkpoint)
            # But check if it's already a PEFT model to avoid double-wrapping
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "PEFT is not installed. Install with: pip install peft"
                )

            # Check if model is already PEFT-wrapped
            from peft import PeftModel
            if not isinstance(model, PeftModel):
                model = self._apply_peft(model)
                print(f"Applied PEFT ({self.config.model.peft_method}) to checkpoint-loaded model")
                self._print_trainable_parameters(model)
            else:
                print("Model already has PEFT adapters, skipping PEFT application")
                self._print_trainable_parameters(model)

        model_class = MODEL_REGISTRY[self.config.model.type]
        model_config = self.config.model

        if model_config.type == "traditional":
            return model_class(
                model=model,
                tokenizer=tokenizer,
                learning_rate=model_config.learning_rate,
                buffer_size=model_config.buffer_size,
                local_sample_limit=model_config.local_sample_limit,
            )
        elif model_config.type == "cmaml":
            return model_class(
                model=model,
                tokenizer=tokenizer,
                inner_loop_learning_rate=model_config.inner_loop_learning_rate,
                learning_rate=model_config.learning_rate,
                buffer_size=model_config.buffer_size,
                local_sample_limit=model_config.local_sample_limit,
            )
        elif model_config.type == "tmaml":
            return model_class(
                model=model,
                tokenizer=tokenizer,
                inner_loop_learning_rate=model_config.inner_loop_learning_rate,
                learning_rate=model_config.learning_rate,
                buffer_size=model_config.buffer_size,
                future_step=model_config.future_step,
                local_sample_limit=model_config.local_sample_limit,
            )
        elif model_config.type == "lamaml":
            return model_class(
                model=model,
                tokenizer=tokenizer,
                nu_lr=model_config.nu_lr,
                alpha_0=model_config.alpha_0,
                buffer_size=model_config.buffer_size,
                local_sample_limit=model_config.local_sample_limit,
            )
        else:
            raise ValueError(f"Unknown model type: {model_config.type}")

    def _create_datamodule(
        self,
        tokenizer: Any,
        train_years: Optional[List[int]] = None,
    ) -> TemporalDataModule:
        """Create data module from config."""
        if train_years is None:
            train_years = self.config.data.train_years

        return TemporalDataModule(
            data_dir=self.config.paths.data_dir,
            train_years=train_years,
            val_years=self.config.data.validation_years,
            test_years=self.config.data.test_years,
            tokenizer=tokenizer,
            batch_size=self.config.training.batch_size,
            max_len=self.config.data.max_len,
            num_workers=self.config.data.num_workers,
            train_sequentially=self.config.data.sequential_training,
            validate_sequentially=self.config.data.sequential_validation,
            test_sequentially=self.config.data.sequential_test,
        )

    def _create_logger(self, year: Optional[int] = None) -> Optional[WandbLogger]:
        """Create WandB logger with experiment config."""
        if not self.config.wandb.enabled:
            return None

        name = f"{self.config.experiment_name}"
        if year:
            name += f"-{year}"
        name += f"-seed-{self.config.seed}"

        return WandbLogger(
            project=self.config.wandb.project,
            group=self.config.experiment_name,
            save_dir=self.config.paths.wandb_dir,
            name=name,
            config=self.config.to_dict(),
        )

    def _create_callbacks(
        self,
        monitor: str,
        year: Optional[int] = None,
    ) -> List[pl.Callback]:
        """Create standard callbacks for checkpointing and early stopping."""
        checkpoint_dir = (
            Path(self.config.paths.checkpoints_dir)
            / self.config.wandb.project
            / self.config.experiment_name
            / f"seed-{self.config.seed}"
        )
        ensure_dir(checkpoint_dir)

        filename = "best-checkpoint"
        if year:
            filename += f"-{year}"

        return [
            pl.callbacks.ModelCheckpoint(
                monitor=monitor,
                dirpath=str(checkpoint_dir),
                filename=filename,
                save_top_k=1,
                mode="max",
            ),
            pl.callbacks.EarlyStopping(
                monitor=monitor,
                patience=self.config.training.early_stopping_patience,
                mode="max",
            ),
        ]

    def _load_from_checkpoint(self, tokenizer: Any) -> BaseModule:
        """Load model from initialization checkpoint."""
        pattern = self.config.initialization.checkpoint_pattern
        checkpoint_path = pattern.format(
            checkpoints_dir=self.config.paths.checkpoints_dir,
            project=self.config.wandb.project,
            seed=self.config.seed,
        )

        print(f"Loading checkpoint from: {checkpoint_path}")
        trad_module = TraditionalModule.load_from_checkpoint(checkpoint_path)

        # Create new module with loaded model
        module = self._create_model(model=trad_module.model, tokenizer=tokenizer)

        # Transfer replay buffer state
        module.replay_buffer.global_samples_seen = trad_module.replay_buffer.global_samples_seen
        module.replay_buffer.data = trad_module.replay_buffer.data

        print(f"Global samples seen: {module.replay_buffer.global_samples_seen}")
        print(f"Buffer size: {len(module.replay_buffer.data)}")

        return module

    def _get_checkpoint_dir(self) -> Path:
        """Get the checkpoint directory for the current experiment."""
        return (
            Path(self.config.paths.checkpoints_dir)
            / self.config.wandb.project
            / self.config.experiment_name
            / f"seed-{self.config.seed}"
        )

    def _get_results_path(self) -> Path:
        """Get the results file path for the current experiment."""
        results_dir = (
            Path(self.config.paths.results_dir)
            / self.config.wandb.project
            / self.config.experiment_name
        )
        return results_dir / f"{self.config.experiment_name}-seed-{self.config.seed}.csv"

    def _find_last_completed_year(self) -> Optional[int]:
        """
        Find the last completed training year by checking for existing checkpoints.

        Returns:
            The last year that has a completed checkpoint, or None if no checkpoints exist.
        """
        checkpoint_dir = self._get_checkpoint_dir()
        if not checkpoint_dir.exists():
            return None

        completed_years = []
        for year in self.config.data.train_years:
            checkpoint_path = checkpoint_dir / f"best-checkpoint-{year}.ckpt"
            if checkpoint_path.exists():
                completed_years.append(year)

        if not completed_years:
            return None

        return max(completed_years)

    def _load_existing_results(self) -> List[Dict[str, Any]]:
        """
        Load existing results from a previous run for resume functionality.

        Returns:
            List of result dictionaries, or empty list if no results exist.
        """
        results_path = self._get_results_path()
        if not results_path.exists():
            return []

        try:
            df = pd.read_csv(results_path)
            return df.to_dict("records")
        except Exception as e:
            print(f"Warning: Could not load existing results from {results_path}: {e}")
            return []

    def _save_results(
        self,
        all_results: List[Dict[str, Any]],
    ) -> None:
        """Save experiment results to CSV."""
        results_dir = (
            Path(self.config.paths.results_dir)
            / self.config.wandb.project
            / self.config.experiment_name
        )
        ensure_dir(results_dir)

        filename = f"{self.config.experiment_name}-seed-{self.config.seed}.csv"
        df = pd.DataFrame(all_results)
        df.to_csv(results_dir / filename, index=False)
        print(f"Saved results to: {results_dir / filename}")

    def run(self) -> None:
        """Run the experiment based on configuration."""
        # TMAML requires sequential mode (needs paired current/future batches)
        if self.config.model.type == "tmaml" and not self.config.training.sequential:
            raise ValueError(
                "TMAML requires sequential training mode (training.sequential: true). "
                "Non-sequential mode does not provide paired current/future batches."
            )

        if self.config.training.sequential:
            self.run_sequential()
        else:
            self.run_nonsequential()

    def run_nonsequential(self) -> None:
        """Run non-sequential training (single training phase)."""
        tokenizer = self._load_tokenizer()
        module = self._create_model(tokenizer=tokenizer)
        dm = self._create_datamodule(tokenizer)
        logger = self._create_logger()
        callbacks = self._create_callbacks(monitor="val_auc")

        trainer = pl.Trainer(
            max_epochs=self.config.training.max_epochs,
            devices=self.config.training.devices,
            accelerator=self.config.training.accelerator,
            logger=logger,
            callbacks=callbacks,
            enable_progress_bar=self.config.training.progress_bar,
        )

        trainer.fit(module, dm)
        dm_results = trainer.test(ckpt_path="best", datamodule=dm)

        # Process and save results
        all_results = []
        for metrics in dm_results:
            for key, value in metrics.items():
                if key.startswith("test_auc/dataloader_idx_"):
                    test_year = self.config.data.test_years[int(key.split("_")[-1])]
                    all_results.append({
                        "train_year": "pooled",
                        "test_year": test_year,
                        "test_auc": value,
                    })

        self._save_results(all_results)

        if self.config.wandb.enabled:
            wandb.finish()

    def run_sequential(self) -> None:
        """Run sequential training (year-by-year)."""
        tokenizer = self._load_tokenizer()

        # Handle resume from last checkpoint
        start_year_idx = 0
        all_results = []

        if self.config.training.resume:
            last_completed_year = self._find_last_completed_year()
            if last_completed_year is not None:
                # Find the index of the next year to train
                try:
                    last_completed_idx = self.config.data.train_years.index(last_completed_year)
                    start_year_idx = last_completed_idx + 1

                    if start_year_idx >= len(self.config.data.train_years):
                        print(f"All years already completed. Last completed year: {last_completed_year}")
                        return

                    # Load the checkpoint from the last completed year
                    checkpoint_path = (
                        self._get_checkpoint_dir() / f"best-checkpoint-{last_completed_year}.ckpt"
                    )
                    print(f"Resuming from year {last_completed_year}, checkpoint: {checkpoint_path}")

                    # Load existing results to preserve them
                    all_results = self._load_existing_results()
                    # Filter to only include results from completed years
                    completed_years = self.config.data.train_years[:start_year_idx]
                    all_results = [
                        r for r in all_results if r.get("train_year") in completed_years
                    ]
                    print(f"Loaded {len(all_results)} existing results from previous run")

                    # Load model from checkpoint
                    model_class = MODEL_REGISTRY[self.config.model.type]
                    module = model_class.load_from_checkpoint(checkpoint_path)
                    print(f"Resuming training from year {self.config.data.train_years[start_year_idx]}")

                except ValueError:
                    print(f"Warning: Last completed year {last_completed_year} not in train_years, starting fresh")
                    start_year_idx = 0
            else:
                print("Resume enabled but no completed checkpoints found, starting fresh")

        # Load from initialization checkpoint if specified and not resuming
        if start_year_idx == 0:
            if self.config.initialization.from_checkpoint:
                module = self._load_from_checkpoint(tokenizer)
            else:
                module = self._create_model(tokenizer=tokenizer)

        for year_idx, year in enumerate(self.config.data.train_years):
            # Skip already completed years when resuming
            if year_idx < start_year_idx:
                continue
            # Reset local sample counter
            module.replay_buffer.reset_local_counter()

            if year not in self.config.data.validation_years:
                raise ValueError(
                    f"Training year {year} not found in validation_years. "
                    f"Train years must be a subset of validation years for monitoring."
                )
            val_idx = self.config.data.validation_years.index(year)
            monitor = f"val_auc/dataloader_idx_{val_idx}"

            logger = self._create_logger(year)

            # TMAML needs paired batches: current year + future year
            if self.config.model.type == "tmaml":
                future_idx = year_idx + self.config.model.future_step
                if future_idx >= len(self.config.data.train_years):
                    print(
                        f"Skipping year {year}: TMAML requires future year at index "
                        f"{future_idx}, but only {len(self.config.data.train_years)} "
                        "training years are available."
                    )
                    break

                future_year = self.config.data.train_years[future_idx]

                dm_current = self._create_datamodule(tokenizer, train_years=[year])
                dm_future = self._create_datamodule(tokenizer, train_years=[future_year])

                dm_current.setup("fit")
                dm_future.setup("fit")

                current_loader = dm_current.train_dataloader()[0]
                future_loader = dm_future.train_dataloader()[0]
                train_loader = CombinedLoader(
                    {"current": current_loader, "future": future_loader},
                    mode="min_size",
                )

                dm = dm_current
                val_dataloaders = dm_current.val_dataloader()
            else:
                dm = self._create_datamodule(tokenizer, train_years=[year])
                val_dataloaders = None
            callbacks = self._create_callbacks(monitor, year)

            trainer = pl.Trainer(
                max_epochs=self.config.training.max_epochs,
                devices=self.config.training.devices,
                accelerator=self.config.training.accelerator,
                logger=logger,
                callbacks=callbacks,
                enable_progress_bar=self.config.training.progress_bar,
            )

            if self.config.model.type == "tmaml":
                trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_dataloaders)
            else:
                trainer.fit(module, dm)
            dm_results = trainer.test(ckpt_path="best", datamodule=dm)

            # Load best checkpoint for next year
            best_checkpoint_path = callbacks[0].best_model_path
            module = type(module).load_from_checkpoint(best_checkpoint_path)

            # Log and collect results
            if logger and hasattr(logger, "experiment"):
                run = logger.experiment
                run.define_metric("test_auc", step_metric="test_year")

            for metrics in dm_results:
                for key, value in metrics.items():
                    if key.startswith("test_auc/dataloader_idx_"):
                        test_year = self.config.data.test_years[int(key.split("_")[-1])]
                        all_results.append({
                            "train_year": year,
                            "test_year": test_year,
                            "test_auc": value,
                        })

                        if logger and hasattr(logger, "experiment"):
                            logger.experiment.log({
                                "test_auc": value,
                                "test_year": test_year,
                            })

            if self.config.wandb.enabled:
                wandb.finish()

            del trainer, dm

        self._save_results(all_results)


def main() -> None:
    """CLI entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run LA-MAML clinical experiments")
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to experiment config file or name (without .yaml)",
    )
    parser.add_argument(
        "--paths", "-p",
        type=str,
        default="gpfs",
        choices=["gpfs", "local"],
        help="Path configuration to use",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        required=True,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Override WandB project name",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)

    args = parser.parse_args()

    # Load and merge configurations
    config = load_config(
        experiment_config=args.config,
        paths_config=args.paths,
        overrides={
            "seed": args.seed,
            "wandb.project": args.project_name,
            "training.batch_size": args.batch_size,
            "training.max_epochs": args.max_epochs,
            "model.learning_rate": args.learning_rate,
        },
    )

    # Create and run experiment
    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
