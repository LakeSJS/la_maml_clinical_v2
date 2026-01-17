"""Result visualization and analysis for LA-MAML experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CI_Z = 1.96  # 95% confidence interval

PLOT_COLORS = [
    "darkorange", "gold", "yellowgreen", "forestgreen", "turquoise",
    "deepskyblue", "dodgerblue", "royalblue", "mediumpurple", "purple",
    "magenta", "deeppink", "brown",
]

FIG_SIZE = (10, 6)
BAR_WIDTH = 0.2


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics grouped by test year.

    Args:
        df: DataFrame with test_year and test_auc columns

    Returns:
        DataFrame with mean, std, count, and confidence interval per year
    """
    if df.empty:
        return pd.DataFrame(columns=["test_year", "mean", "std", "count", "ci"])

    stats = (
        df.groupby("test_year")["test_auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("test_year")
    )
    stats["ci"] = np.where(
        stats["count"] > 0,
        CI_Z * stats["std"] / np.sqrt(stats["count"]),
        np.nan,
    )
    return stats


def add_line(
    ax: plt.Axes,
    stats: pd.DataFrame,
    color: str,
    label: str,
    marker: str = "o",
    linestyle: str = "-",
) -> None:
    """
    Add a line with confidence interval to a plot.

    Args:
        ax: Matplotlib axes
        stats: Summary statistics DataFrame
        color: Line color
        label: Legend label
        marker: Marker style
        linestyle: Line style
    """
    if stats.empty:
        print(f"Skipping {label} because no data available.")
        return

    lower = stats["mean"] - stats["ci"]
    upper = stats["mean"] + stats["ci"]
    ax.fill_between(
        stats["test_year"], lower, upper, alpha=0.2, color=color, label="_nolegend_"
    )
    ax.plot(
        stats["test_year"],
        stats["mean"],
        marker=marker,
        linestyle=linestyle,
        color=color,
        label=label,
    )


def per_seed_differences(
    baseline_df: pd.DataFrame,
    method_df: pd.DataFrame,
) -> Tuple[List[float], set]:
    """
    Compute per-seed differences between baseline and method.

    Args:
        baseline_df: Baseline results DataFrame
        method_df: Method results DataFrame

    Returns:
        Tuple of (list of differences, set of missing seeds)
    """
    baseline_values = baseline_df["test_auc"].values
    baseline_seeds = baseline_df["seed"].values
    method_values = method_df["test_auc"].values
    method_seeds = method_df["seed"].values

    common = set(baseline_seeds) & set(method_seeds)
    baseline_lookup = dict(zip(baseline_seeds, baseline_values))
    method_lookup = dict(zip(method_seeds, method_values))

    diffs = [method_lookup[seed] - baseline_lookup[seed] for seed in common]
    missing = (set(baseline_seeds) | set(method_seeds)) - common

    return diffs, missing


def summarize_differences(diffs: Iterable[float]) -> Tuple[float, float]:
    """
    Compute mean and confidence interval for differences.

    Args:
        diffs: Iterable of difference values

    Returns:
        Tuple of (mean, confidence interval)
    """
    diffs = list(diffs)
    if not diffs:
        return float("nan"), float("nan")

    arr = np.asarray(diffs, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    ci = CI_Z * std / np.sqrt(len(arr)) if len(arr) else float("nan")
    return mean, ci


def load_results(
    results_dir: Path,
    experiment_folders: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Load and aggregate results from multiple experiments.

    Args:
        results_dir: Root directory containing experiment results
        experiment_folders: Optional mapping of experiment names to folder names

    Returns:
        Aggregated DataFrame with all results
    """
    if experiment_folders is None:
        experiment_folders = {
            "trad-nonseq-2013-2024": "traditional-nonsequential-2013-2024",
            "trad-nonseq-2013-2018": "traditional-nonsequential-2013-2018",
            "trad-seq-2019-2024": "traditional-sequential-2019-2024",
            "cmaml-seq-2019-2024": "cmaml-sequential-2019-2024",
            "lamaml-seq-2019-2024": "lamaml-sequential-2019-2024",
        }

    frames = []
    for experiment, folder_name in experiment_folders.items():
        folder_path = results_dir / folder_name
        if not folder_path.exists():
            print(f"Warning: {folder_path} does not exist for experiment {experiment}")
            continue

        csv_paths = sorted(p for p in folder_path.iterdir() if p.suffix == ".csv")
        if not csv_paths:
            print(f"No result files found in {folder_path}")
            continue

        for csv_path in csv_paths:
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"Error processing {csv_path}: {exc}")
                continue

            df = df.copy()
            df["experiment"] = experiment
            df["seed"] = csv_path.stem.split("-")[-1]
            if "train_year" in df.columns:
                df["train_year"] = df["train_year"].astype(str)
            frames.append(df)

    if not frames:
        raise RuntimeError("No result files found for any experiment.")

    results_df = pd.concat(frames, ignore_index=True).drop_duplicates()
    if "test_year" in results_df.columns:
        results_df["test_year"] = pd.to_numeric(results_df["test_year"], errors="coerce")

    return results_df


def make_temporal_falloff_plot(
    results_df: pd.DataFrame,
    sequential_key: str,
    title: str,
    ylim: Tuple[float, float] = (0.79, 0.865),
) -> plt.Figure:
    """
    Create temporal falloff plot showing performance over test years.

    Args:
        results_df: Aggregated results DataFrame
        sequential_key: Key for the sequential experiment to highlight
        title: Plot title
        ylim: Y-axis limits

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.set_title(title)

    # Foundation model (trained on all data)
    foundation_df = results_df[results_df["experiment"] == "trad-nonseq-2013-2024"]
    add_line(ax, compute_summary(foundation_df), "blue", "Train year: 2013-2024")

    # Baseline model (trained on 2013-2018)
    baseline_df = results_df[results_df["experiment"] == "trad-nonseq-2013-2018"]
    add_line(ax, compute_summary(baseline_df), "red", "Train year: 2013-2018")

    # Sequential experiments
    sequential_df = results_df[results_df["experiment"] == sequential_key]
    train_years = sequential_df["train_year"].dropna().unique()

    for idx, train_year in enumerate(train_years):
        year_df = sequential_df[sequential_df["train_year"] == train_year]
        color = PLOT_COLORS[idx % len(PLOT_COLORS)]
        add_line(ax, compute_summary(year_df), color, f"Finetuned through: {train_year}")

    ax.set_xlabel("Test Year")
    ax.set_ylabel("Test AUC")
    ax.set_xticks(np.arange(2013, 2025, 1))
    ax.set_ylim(*ylim)
    ax.legend()
    fig.tight_layout()

    return fig


def save_plot(fig: plt.Figure, output_path: Path) -> None:
    """Save figure to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main() -> None:
    """Generate all evaluation plots."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate LA-MAML evaluation plots")
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save plots",
    )

    args = parser.parse_args()

    results_df = load_results(args.results_dir)

    # Generate plots
    plots = [
        (
            make_temporal_falloff_plot(
                results_df,
                "trad-seq-2019-2024",
                "Temporal Falloff - Traditional Sequential",
            ),
            "temporal_falloff_traditional.png",
        ),
        (
            make_temporal_falloff_plot(
                results_df,
                "cmaml-seq-2019-2024",
                "Temporal Falloff - CMAML Sequential",
            ),
            "temporal_falloff_cmaml.png",
        ),
        (
            make_temporal_falloff_plot(
                results_df,
                "lamaml-seq-2019-2024",
                "Temporal Falloff - LA-MAML Sequential",
            ),
            "temporal_falloff_lamaml.png",
        ),
    ]

    for fig, filename in plots:
        save_plot(fig, args.output_dir / filename)


if __name__ == "__main__":
    main()
