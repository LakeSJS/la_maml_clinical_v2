"""Integration tests for ExperimentRunner."""

import pytest
from pathlib import Path

from lamaml_clinical.core.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PathsConfig,
    TrainingConfig,
    WandbConfig,
)
from lamaml_clinical.training.runner import ExperimentRunner, MODEL_REGISTRY


class TestModelRegistry:
    """Test model registry."""

    def test_all_models_registered(self):
        """Test that all model types are in registry."""
        assert "traditional" in MODEL_REGISTRY
        assert "cmaml" in MODEL_REGISTRY
        assert "lamaml" in MODEL_REGISTRY

    def test_registry_returns_correct_classes(self):
        """Test that registry returns correct module classes."""
        from lamaml_clinical.models import (
            TraditionalModule,
            CmamlModule,
            LamamlModule,
        )

        assert MODEL_REGISTRY["traditional"] is TraditionalModule
        assert MODEL_REGISTRY["cmaml"] is CmamlModule
        assert MODEL_REGISTRY["lamaml"] is LamamlModule


class TestExperimentRunner:
    """Test ExperimentRunner initialization and configuration."""

    @pytest.fixture
    def minimal_config(self, tmp_path):
        """Create minimal test configuration."""
        return ExperimentConfig(
            experiment_name="test-experiment",
            seed=42,
            paths=PathsConfig(
                base_model_id="distilbert-base-uncased",
                tokenizer_path="distilbert-base-uncased",
                data_dir=str(tmp_path / "data"),
                checkpoints_dir=str(tmp_path / "checkpoints"),
                results_dir=str(tmp_path / "results"),
                wandb_dir=str(tmp_path / "wandb"),
            ),
            data=DataConfig(
                train_years=[2019],
                validation_years=[2019],
                test_years=[2019],
                max_len=64,
            ),
            model=ModelConfig(
                type="traditional",
                learning_rate=1e-3,
                buffer_size=5,
            ),
            training=TrainingConfig(
                batch_size=2,
                max_epochs=1,
                sequential=False,
            ),
            wandb=WandbConfig(
                enabled=False,  # Disable for testing
            ),
        )

    def test_runner_initialization(self, minimal_config):
        """Test that runner initializes without error."""
        runner = ExperimentRunner(minimal_config)
        assert runner.config.seed == 42

    def test_runner_creates_model(self, minimal_config):
        """Test that runner can create model."""
        runner = ExperimentRunner(minimal_config)
        tokenizer = runner._load_tokenizer()
        model = runner._create_model(tokenizer=tokenizer)

        from lamaml_clinical.models import TraditionalModule
        assert isinstance(model, TraditionalModule)

    def test_runner_creates_datamodule(self, minimal_config):
        """Test that runner can create datamodule."""
        runner = ExperimentRunner(minimal_config)
        tokenizer = runner._load_tokenizer()
        dm = runner._create_datamodule(tokenizer)

        from lamaml_clinical.data import TemporalDataModule
        assert isinstance(dm, TemporalDataModule)
