#!/usr/bin/env python3
"""
Unified CLI entry point for LA-MAML clinical experiments.

Examples:
    # Traditional non-sequential training
    python scripts/run_experiment.py \
        --config traditional_nonseq_2013_2018 \
        --paths gpfs \
        --seed 42

    # LA-MAML sequential training
    python scripts/run_experiment.py \
        --config lamaml_seq_2019_2024 \
        --paths gpfs \
        --seed 42

    # Local development with custom settings
    python scripts/run_experiment.py \
        --config traditional_nonseq_2013_2018 \
        --paths local \
        --seed 42 \
        --max-epochs 1
"""

import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).resolve().parents[1] / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from lamaml_clinical.training.runner import main

if __name__ == "__main__":
    main()
