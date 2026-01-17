#!/bin/bash
#SBATCH --job-name=lamaml_exp
#SBATCH --output=%x_%j.log
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

# =============================================================================
# Unified SLURM submission script for LA-MAML clinical experiments
#
# Usage:
#   sbatch --export=CONFIG=lamaml_seq_2019_2024,SEED=42 scripts/slurm/submit.sh
#
# Required environment variables (pass via --export):
#   CONFIG    - Experiment config name (without .yaml) or path to config file
#   SEED      - Random seed for reproducibility
#
# Optional environment variables:
#   PATHS        - Path config to use (gpfs, local). Default: gpfs
#   PROJECT_NAME - Override WandB project name
#   CONDA_ENV    - Conda environment name. Default: amazon_fashion_env
#
# To override SLURM settings, use sbatch flags:
#   sbatch --partition=oermannlab --mem=256G --time=10-00:00:00 \
#          --export=CONFIG=lamaml_seq_2019_2024,SEED=42 scripts/slurm/submit.sh
#
# To redirect logs to a specific directory:
#   sbatch --output=/path/to/logs/%x_%j.log \
#          --export=CONFIG=lamaml_seq_2019_2024,SEED=42 scripts/slurm/submit.sh
#
# Examples:
#   # Basic submission
#   sbatch --export=CONFIG=lamaml_seq_2019_2024,SEED=42 scripts/slurm/submit.sh
#
#   # With custom partition and memory
#   sbatch --partition=oermannlab --mem=256G \
#          --export=CONFIG=lamaml_seq_2019_2024,SEED=42 scripts/slurm/submit.sh
#
#   # Run all seeds for an experiment
#   for seed in 1 2 3 4 5; do
#       sbatch --export=CONFIG=cmaml_seq_2019_2024,SEED=$seed scripts/slurm/submit.sh
#   done
#
#   # Custom job name
#   sbatch --job-name=lamaml_seed42 \
#          --export=CONFIG=lamaml_seq_2019_2024,SEED=42 scripts/slurm/submit.sh
# =============================================================================

set -e

# Print job info
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Start time: $(date)"
echo "Config: ${CONFIG}"
echo "Seed: ${SEED}"
echo "Paths: ${PATHS:-gpfs}"
echo "============================================"

# Activate conda environment
source ~/.bashrc
conda activate "${CONDA_ENV:-amazon_fashion_env}"

# Enable CUDA DSA for better error messages
export TORCH_USE_CUDA_DSA=1

# Find project root (script is in scripts/slurm/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Load environment variables from .env if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment from $PROJECT_ROOT/.env"
    set -a  # automatically export all variables
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Validate required variables
if [ -z "$CONFIG" ]; then
    echo "ERROR: CONFIG environment variable is required"
    echo "Usage: sbatch --export=CONFIG=<config_name>,SEED=<seed> scripts/slurm/submit.sh"
    exit 1
fi

if [ -z "$SEED" ]; then
    echo "ERROR: SEED environment variable is required"
    echo "Usage: sbatch --export=CONFIG=<config_name>,SEED=<seed> scripts/slurm/submit.sh"
    exit 1
fi

# Build command
CMD="python $PROJECT_ROOT/scripts/run_experiment.py"
CMD="$CMD --config ${CONFIG}"
CMD="$CMD --paths ${PATHS:-gpfs}"
CMD="$CMD --seed ${SEED}"

# Add optional overrides
if [ -n "$PROJECT_NAME" ]; then
    CMD="$CMD --project-name ${PROJECT_NAME}"
fi

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch-size ${BATCH_SIZE}"
fi

if [ -n "$MAX_EPOCHS" ]; then
    CMD="$CMD --max-epochs ${MAX_EPOCHS}"
fi

if [ -n "$LEARNING_RATE" ]; then
    CMD="$CMD --learning-rate ${LEARNING_RATE}"
fi

echo "Running: $CMD"
echo "============================================"

# Run the experiment
$CMD

echo "============================================"
echo "End time: $(date)"
echo "Job completed successfully"
