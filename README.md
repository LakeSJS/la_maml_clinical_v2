# LA-MAML Clinical

Continual meta-learning for clinical NLP with temporal distribution shift mitigation.

## Overview

This repository implements continual learning approaches for 30-day hospital readmission prediction from clinical discharge notes, addressing temporal distribution shift in clinical data.

### Methods

- **Traditional Fine-tuning**: Standard supervised learning baseline
- **CMAML**: Continual Model-Agnostic Meta-Learning with experience replay
- **LA-MAML**: Learning-rate Adaptive MAML with per-parameter learnable learning rates

## Installation

```bash
# Install in development mode
pip install -e ".[dev]"

# Install with plotting dependencies
pip install -e ".[dev,plotting]"
```

## Quick Start

### Running Experiments

```bash
# Traditional non-sequential training (baseline)
python scripts/run_experiment.py \
    --config traditional_nonseq_2013_2018 \
    --paths gpfs \
    --seed 42

# LA-MAML sequential training
python scripts/run_experiment.py \
    --config lamaml_seq_2019_2024 \
    --paths gpfs \
    --seed 42

# Local development (with reduced epochs)
python scripts/run_experiment.py \
    --config traditional_nonseq_2013_2018 \
    --paths local \
    --seed 42 \
    --max-epochs 1
```

### SLURM Submission

```bash
# Submit single experiment
sbatch --export=CONFIG=lamaml_seq_2019_2024,SEED=42 scripts/slurm/submit.sh

# Submit multiple seeds
for seed in 1 2 3 4 5; do
    sbatch --export=CONFIG=lamaml_seq_2019_2024,SEED=$seed scripts/slurm/submit.sh
done

# With custom resources
sbatch --export=CONFIG=lamaml_seq_2019_2024,SEED=42,MEMORY=256G,PARTITION=oermannlab \
       scripts/slurm/submit.sh
```

## Project Structure

```
la_maml_clinical_v2/
├── configs/                      # YAML configuration files
│   ├── base.yaml                 # Default values
│   ├── paths/                    # Environment-specific paths
│   │   ├── gpfs.yaml             # NYU HPC cluster
│   │   └── local.yaml            # Local development
│   ├── experiments/              # Experiment configurations
│   │   ├── traditional_nonseq_2013_2018.yaml
│   │   ├── cmaml_seq_2019_2024.yaml
│   │   └── lamaml_seq_2019_2024.yaml
│   └── models/                   # Model-specific hyperparameters
│       ├── traditional.yaml
│       ├── cmaml.yaml
│       └── lamaml.yaml
│
├── src/lamaml_clinical/          # Main package
│   ├── core/                     # Configuration and paths
│   ├── data/                     # Data loading
│   ├── models/                   # Model implementations
│   ├── training/                 # Training infrastructure
│   ├── evaluation/               # Plotting and analysis
│   └── utils/                    # Utilities
│
├── scripts/
│   ├── run_experiment.py         # Unified CLI
│   └── slurm/submit.sh           # SLURM submission script
│
└── tests/                        # Test suite
```

## Configuration

### Environment Variables

Override paths via environment variables:

```bash
export NYUTRON_MODEL_DIR="/path/to/model"
export NYUTRON_TOKENIZER_DIR="/path/to/tokenizer"
export LAMAML_DATA_DIR="/path/to/data"
export LAMAML_CHECKPOINTS_DIR="/path/to/checkpoints"
export LAMAML_RESULTS_DIR="/path/to/results"
export WANDB_DIR="/path/to/wandb"
```

### CLI Overrides

Override any config value via CLI:

```bash
python scripts/run_experiment.py \
    --config lamaml_seq_2019_2024 \
    --seed 42 \
    --batch-size 8 \
    --max-epochs 5 \
    --learning-rate 2e-5
```

## Models

### TraditionalModule

Standard fine-tuning with AdamW optimizer. Populates replay buffer for use by meta-learning modules.

### CmamlModule

Continual MAML with:
- Inner loop: One-sample-at-a-time gradient descent
- Outer loop: Meta-optimization on current + replay buffer samples

### LamamlModule

LA-MAML with per-parameter learnable learning rates:
- Each model parameter has its own learnable learning rate
- Two optimizers: meta-optimizer (model weights) + LR optimizer (learning rates)
- Gradient clipping for stability

## PEFT (Parameter-Efficient Fine-Tuning) Support

This project now supports PEFT methods for memory and compute-efficient training. PEFT integration is **completely transparent** to the training loops - it wraps the base model at initialization time.

### Enabling PEFT

Add PEFT configuration to your model config file or use CLI overrides:

```yaml
model:
  use_peft: true
  peft_method: lora      # Currently only LoRA is supported
  lora_r: 8              # LoRA rank (lower = fewer parameters)
  lora_alpha: 16         # LoRA scaling factor
  lora_dropout: 0.1
  lora_target_modules: ["query", "value"]  # null for auto-detect
  lora_bias: none        # Options: none, all, lora_only
```

### Example Usage

**Option 1: Modify experiment config**

Edit an experiment config file to enable PEFT:

```yaml
# configs/experiments/lamaml_seq_2019_2024_peft.yaml
model:
  type: lamaml
  use_peft: true
  lora_r: 8
  lora_alpha: 16
```

Then run:
```bash
python scripts/run_experiment.py \
    --config lamaml_seq_2019_2024_peft \
    --paths gpfs \
    --seed 42
```

**Option 2: Use PEFT model config templates**

Copy one of the provided PEFT model configs (`configs/models/*_peft.yaml`) to your experiment config:

```yaml
# In your experiment config file
model:
  <<: !include ../models/lamaml_peft.yaml
```

See `configs/models/lamaml_peft.yaml`, `configs/models/cmaml_peft.yaml`, and `configs/models/traditional_peft.yaml` for complete PEFT configuration templates.

### PEFT Benefits

- **Memory Efficiency**: Train only 1-5% of parameters (typical LoRA configuration)
- **Faster Training**: Fewer parameters to update per step
- **No Training Loop Changes**: PEFT is applied at model initialization; all training logic remains identical
- **Maintains Meta-Learning**: PEFT works seamlessly with MAML inner/outer loops

### PEFT Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_peft` | `false` | Enable/disable PEFT |
| `peft_method` | `lora` | PEFT method (currently only LoRA supported) |
| `lora_r` | `8` | LoRA rank - controls adapter size (typically 4-64) |
| `lora_alpha` | `16` | LoRA scaling factor (typically 2x rank) |
| `lora_dropout` | `0.1` | Dropout probability for LoRA layers |
| `lora_target_modules` | `null` | Which modules to adapt (null = auto-detect for BERT: ["query", "value"]) |
| `lora_bias` | `none` | Bias handling: "none", "all", or "lora_only" |

### Implementation Notes

**What Changed:**
- Added PEFT configuration fields to `ModelConfig` (src/lamaml_clinical/core/config.py:66-73)
- Model wrapping logic in `ExperimentRunner._load_base_model()` (src/lamaml_clinical/training/runner.py:72-172)
- Fixed PEFT compatibility in `_create_model()` for checkpoint initialization (src/lamaml_clinical/training/runner.py:156-172)
- Fixed parameter alignment in `LamamlModule` to match learnable LRs with trainable params (src/lamaml_clinical/models/lamaml.py:173)
- Added `peft>=0.7.0` dependency

**What Didn't Change:**
- All training loop logic remains unchanged (though parameter iteration was fixed to be PEFT-compatible)
- All model forward passes remain unchanged
- Replay buffer logic remains unchanged
- Checkpoint loading/saving works automatically (PEFT state is saved in model weights)

**Critical Fixes for PEFT Compatibility:**

1. **LAMAML Parameter Alignment**: Fixed bug where gradient scaling zipped all parameters with learnable LRs (which only exist for trainable params). Now correctly filters to trainable parameters only.

2. **Checkpoint Initialization with PEFT**: When loading from a checkpoint to initialize a new module type (e.g., Traditional → LAMAML), PEFT adapters are now correctly applied. Includes check to avoid double-wrapping if checkpoint already has PEFT.

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src/lamaml_clinical --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py -v
```

## Generating Plots

```bash
python -m lamaml_clinical.evaluation.plotting \
    --results-dir results/NYUtron-temporal-falloff \
    --output-dir results/plots
```

## Citation

If you use this code, please cite:

```bibtex
@article{lamaml_clinical,
  title={Continual Meta-Learning for Clinical NLP with Temporal Distribution Shift},
  author={Jacobs-Skolik, Spencer},
  year={2024}
}
```

## License

MIT License
