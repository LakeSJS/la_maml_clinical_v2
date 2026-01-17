"""Tests for configuration loading and validation."""

from pathlib import Path

import pytest

from lamaml_clinical.core.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PathsConfig,
    TrainingConfig,
    WandbConfig,
    load_config,
)
from lamaml_clinical.core.paths import resolve_path


class TestPathsConfig:
    """Test PathsConfig dataclass."""

    def test_default_values(self):
        """Test default path configuration."""
        config = PathsConfig()
        assert config.checkpoints_dir == "./checkpoints"
        assert config.results_dir == "./results"

    def test_resolve_all(self, tmp_path):
        """Test path resolution."""
        config = PathsConfig(
            checkpoints_dir=str(tmp_path / "checkpoints"),
            results_dir=str(tmp_path / "results"),
        )
        resolved = config.resolve_all()
        assert Path(resolved.checkpoints_dir).is_absolute()


class TestDataConfig:
    """Test DataConfig dataclass."""

    def test_default_values(self):
        """Test default data configuration."""
        config = DataConfig()
        assert config.max_len == 512
        assert config.sequential_training is True

    def test_custom_years(self):
        """Test custom year configuration."""
        config = DataConfig(
            train_years=[2019, 2020],
            validation_years=[2019, 2020],
            test_years=[2013, 2014, 2015],
        )
        assert len(config.train_years) == 2
        assert len(config.test_years) == 3


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_default_type(self):
        """Test default model type."""
        config = ModelConfig()
        assert config.type == "traditional"
        assert config.buffer_size == 500

    def test_lamaml_params(self):
        """Test LA-MAML specific parameters."""
        config = ModelConfig(
            type="lamaml",
            alpha_0=1e-5,
            nu_lr=1e-6,
        )
        assert config.alpha_0 == 1e-5
        assert config.nu_lr == 1e-6


class TestExperimentConfig:
    """Test ExperimentConfig dataclass."""

    def test_default_experiment(self):
        """Test default experiment configuration."""
        config = ExperimentConfig()
        assert config.experiment_name == "experiment"
        assert config.seed == 42

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ExperimentConfig(experiment_name="test-exp", seed=123)
        config_dict = config.to_dict()
        assert config_dict["experiment_name"] == "test-exp"
        assert config_dict["seed"] == 123


class TestResolvePath:
    """Test path resolution utilities."""

    def test_expand_user(self):
        """Test user directory expansion."""
        path = resolve_path("~/test")
        assert "~" not in str(path)

    def test_absolute_path(self):
        """Test absolute path handling."""
        path = resolve_path("/absolute/path")
        assert path.is_absolute()
