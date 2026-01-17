"""Monitoring utilities for LA-MAML training."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch


def compute_lr_statistics(learnable_lrs: torch.nn.ParameterList) -> Dict[str, float]:
    """
    Compute statistics for learnable learning rates.

    Args:
        learnable_lrs: ParameterList of learnable learning rates

    Returns:
        Dictionary with statistics (mean, std, min, max, etc.)
    """
    lr_values = torch.cat([lr.flatten().detach() for lr in learnable_lrs])

    return {
        "lr_mean": lr_values.mean().item(),
        "lr_std": lr_values.std().item(),
        "lr_min": lr_values.min().item(),
        "lr_max": lr_values.max().item(),
        "lr_median": lr_values.median().item(),
        "lr_q25": lr_values.quantile(0.25).item(),
        "lr_q75": lr_values.quantile(0.75).item(),
        "lr_num_positive": (lr_values > 0).sum().item(),
        "lr_num_negative": (lr_values < 0).sum().item(),
        "lr_num_total": lr_values.numel(),
    }


def log_lr_statistics(
    learnable_lrs: torch.nn.ParameterList,
    logger: Any,
    step: Optional[int] = None,
) -> None:
    """
    Log learning rate statistics to wandb or other logger.

    Args:
        learnable_lrs: ParameterList of learnable learning rates
        logger: PyTorch Lightning logger (e.g., WandbLogger)
        step: Current step number (optional)
    """
    stats = compute_lr_statistics(learnable_lrs)
    for key, value in stats.items():
        logger.log_metrics({key: value}, step=step)


def check_lr_health(learnable_lrs: torch.nn.ParameterList) -> Dict[str, Any]:
    """
    Check the health of learnable learning rates and flag potential issues.

    Args:
        learnable_lrs: ParameterList of learnable learning rates

    Returns:
        Dictionary with health check results including issues and warnings
    """
    stats = compute_lr_statistics(learnable_lrs)
    issues: List[str] = []
    warnings: List[str] = []

    if stats["lr_num_negative"] > 0:
        issues.append(f"Found {stats['lr_num_negative']} negative learning rates")

    if stats["lr_max"] > 1.0:
        warnings.append(f"Very large learning rates found (max: {stats['lr_max']:.6f})")

    if stats["lr_min"] < 1e-8:
        warnings.append(f"Very small learning rates found (min: {stats['lr_min']:.8f})")

    if np.isnan(stats["lr_mean"]):
        issues.append("NaN learning rates detected")

    if stats["lr_std"] < 1e-8:
        warnings.append("Learning rates have very low variance - may not be adapting")

    return {
        "stats": stats,
        "issues": issues,
        "warnings": warnings,
        "healthy": len(issues) == 0,
    }


def plot_lr_distribution(
    learnable_lrs: torch.nn.ParameterList,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the distribution of learnable learning rates.

    Args:
        learnable_lrs: ParameterList of learnable learning rates
        save_path: Path to save the plot (optional, displays if None)
    """
    import matplotlib.pyplot as plt

    lr_values = torch.cat([lr.flatten().detach() for lr in learnable_lrs]).cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1.hist(lr_values, bins=50, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Learning Rate")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Learnable Learning Rates")
    ax1.grid(True, alpha=0.3)

    # Box plot
    ax2.boxplot(lr_values)
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Box Plot")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()


def monitor_buffer_stats(replay_buffer: Any) -> Dict[str, Any]:
    """
    Monitor replay buffer statistics.

    Args:
        replay_buffer: ReplayBuffer instance

    Returns:
        Dictionary with buffer statistics
    """
    return {
        "buffer_size": len(replay_buffer.data),
        "buffer_capacity": replay_buffer.buffer_size,
        "buffer_utilization": len(replay_buffer.data) / replay_buffer.buffer_size,
        "global_samples_seen": replay_buffer.global_samples_seen,
        "local_samples_seen": replay_buffer.local_samples_seen,
    }
