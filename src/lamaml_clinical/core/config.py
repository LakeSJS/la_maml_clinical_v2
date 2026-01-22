"""Configuration dataclasses and YAML loading utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from omegaconf import OmegaConf

from lamaml_clinical.core.paths import resolve_path


@dataclass
class PathsConfig:
    """Configuration for all file paths."""

    base_model_id: str = ""
    tokenizer_path: str = ""
    data_dir: str = ""
    checkpoints_dir: str = "./checkpoints"
    results_dir: str = "./results"
    wandb_dir: str = "./wandb"
    logs_dir: str = "./logs"

    def resolve_all(self) -> "PathsConfig":
        """Resolve all paths, expanding environment variables."""
        return PathsConfig(
            base_model_id=str(resolve_path(self.base_model_id)),
            tokenizer_path=str(resolve_path(self.tokenizer_path)),
            data_dir=str(resolve_path(self.data_dir)),
            checkpoints_dir=str(resolve_path(self.checkpoints_dir)),
            results_dir=str(resolve_path(self.results_dir)),
            wandb_dir=str(resolve_path(self.wandb_dir)),
            logs_dir=str(resolve_path(self.logs_dir)),
        )


@dataclass
class DataConfig:
    """Configuration for data loading."""

    train_years: List[int] = field(default_factory=list)
    validation_years: List[int] = field(default_factory=list)
    test_years: List[int] = field(default_factory=list)
    max_len: int = 512
    num_workers: int = 0
    sequential_training: bool = True
    sequential_validation: bool = True
    sequential_test: bool = True


@dataclass
class ModelConfig:
    """Configuration for model architecture and hyperparameters."""

    type: str = "traditional"  # traditional, cmaml, tmaml, lamaml
    learning_rate: float = 1e-5
    inner_loop_learning_rate: float = 1e-5  # For CMAML/TMAML
    future_step: int = 1  # For TMAML
    alpha_0: float = 1e-5  # For LAMAML
    nu_lr: float = 1e-6  # For LAMAML
    buffer_size: int = 500
    local_sample_limit: Optional[int] = None
    gradient_clip_norm: float = 2.0

    # PEFT (Parameter-Efficient Fine-Tuning) configuration
    use_peft: bool = False
    peft_method: str = "lora"  # lora, prefix, p-tuning, etc.
    lora_r: int = 8  # LoRA rank
    lora_alpha: int = 16  # LoRA scaling factor
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None  # None = auto-detect for model type
    lora_bias: str = "none"  # none, all, lora_only


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 4
    max_epochs: int = 10
    early_stopping_patience: int = 3
    progress_bar: bool = False
    precision: str = "32"
    accelerator: str = "gpu"
    devices: int = 1
    sequential: bool = True


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""

    project: str = "NYUtron-temporal-falloff"
    group: Optional[str] = None
    enabled: bool = True


@dataclass
class InitializationConfig:
    """Configuration for model initialization."""

    from_checkpoint: bool = False
    checkpoint_pattern: str = ""


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    experiment_name: str = "experiment"
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary for logging."""
        return OmegaConf.to_container(OmegaConf.structured(self), resolve=True)


def load_yaml(path: Path) -> OmegaConf:
    """Load a YAML file using OmegaConf (supports ${oc.env:VAR,default} syntax)."""
    return OmegaConf.load(path)


def load_config(
    experiment_config: str | Path,
    paths_config: str = "gpfs",
    overrides: Optional[dict[str, Any]] = None,
    config_dir: Optional[Path] = None,
) -> ExperimentConfig:
    """
    Load and merge configuration files.

    Args:
        experiment_config: Path to experiment config file or name (without .yaml)
        paths_config: Name of paths config (gpfs, local) or path to file
        overrides: Dictionary of override values (dot notation keys supported)
        config_dir: Base directory for config files (defaults to configs/)

    Returns:
        Merged ExperimentConfig
    """
    if config_dir is None:
        # Find configs directory relative to package or cwd
        config_dir = Path(__file__).parents[3] / "configs"
        if not config_dir.exists():
            config_dir = Path.cwd() / "configs"

    # Start with empty config
    merged_cfg = OmegaConf.create({})

    # Load and merge base config
    base_path = config_dir / "base.yaml"
    if base_path.exists():
        base_cfg = load_yaml(base_path)
        merged_cfg = OmegaConf.merge(merged_cfg, base_cfg)

    # Load and merge paths config
    if not paths_config.endswith(".yaml"):
        paths_path = config_dir / "paths" / f"{paths_config}.yaml"
    else:
        paths_path = Path(paths_config)
    if paths_path.exists():
        paths_cfg = load_yaml(paths_path)
        merged_cfg = OmegaConf.merge(merged_cfg, paths_cfg)

    # Load experiment config first to determine model type
    if not str(experiment_config).endswith(".yaml"):
        exp_path = config_dir / "experiments" / f"{experiment_config}.yaml"
    else:
        exp_path = Path(experiment_config)
    if exp_path.exists():
        exp_cfg = load_yaml(exp_path)
    else:
        exp_cfg = OmegaConf.create({})

    # Load and merge model-specific config as defaults (before experiment config)
    model_type = OmegaConf.select(exp_cfg, "model.type", default="traditional")
    model_config_path = config_dir / "models" / f"{model_type}.yaml"
    if model_config_path.exists():
        model_cfg = load_yaml(model_config_path)
        merged_cfg = OmegaConf.merge(merged_cfg, model_cfg)

    # Now merge experiment config (takes precedence over model defaults)
    merged_cfg = OmegaConf.merge(merged_cfg, exp_cfg)

    # Apply CLI overrides
    if overrides:
        override_cfg = OmegaConf.create({})
        for key, value in overrides.items():
            if value is not None:
                OmegaConf.update(override_cfg, key, value, merge=True)
        merged_cfg = OmegaConf.merge(merged_cfg, override_cfg)

    # Merge with schema and resolve interpolations
    schema = OmegaConf.structured(ExperimentConfig)
    final_cfg = OmegaConf.merge(schema, merged_cfg)

    # Resolve all interpolations (including ${oc.env:...})
    OmegaConf.resolve(final_cfg)

    # Convert to Python object
    config = OmegaConf.to_object(final_cfg)

    # Resolve paths (expand ~ and make absolute)
    config.paths = config.paths.resolve_all()

    return config
