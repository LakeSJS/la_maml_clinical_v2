#!/bin/bash
#SBATCH --job-name=tmaml_test
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/tmaml_test_%j.log

# Activate conda
eval "$('/gpfs/share/apps/miniconda3/gpu/4.9.2/bin/conda' 'shell.bash' 'hook')"
conda activate amazon_fashion_env

cd /gpfs/data/oermannlab/users/slj9342/la_maml_clinical_v2

echo "Running TMAML batch pairing test..."
python scripts/test_tmaml_batches.py

echo "Test completed with exit code: $?"
