"""PyTorch Lightning DataModules for temporal clinical data."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from lamaml_clinical.data.datasets import ReadmissionDataset


class TemporalDataModule(pl.LightningDataModule):
    """
    DataModule for temporal clinical data with year-based splits.

    Supports both sequential (year-by-year) and non-sequential (pooled) training modes.

    Args:
        data_dir: Directory containing parquet files (train_YYYY.parquet, etc.)
        train_years: List of years to include in training
        val_years: List of years to include in validation
        test_years: List of years to include in testing
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for dataloaders
        max_len: Maximum sequence length for tokenization
        num_workers: Number of dataloader workers
        train_sequentially: If True, create separate loader per training year
        validate_sequentially: If True, create separate loader per validation year
        test_sequentially: If True, create separate loader per test year
    """

    def __init__(
        self,
        data_dir: str | Path,
        train_years: List[int],
        val_years: List[int],
        test_years: List[int],
        tokenizer: Any,
        batch_size: int = 32,
        max_len: int = 512,
        num_workers: int = 0,
        train_sequentially: bool = True,
        validate_sequentially: bool = True,
        test_sequentially: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_workers = num_workers
        self.train_sequentially = train_sequentially
        self.validate_sequentially = validate_sequentially
        self.test_sequentially = test_sequentially

        self.train_loaders: List[DataLoader] = []
        self.train_loader_class_weights: List[torch.Tensor] = []
        self.val_loaders: List[DataLoader] = []
        self.test_loaders: List[DataLoader] = []

    def add_new_data(
        self,
        new_data_path: str | Path,
        split_ratios: List[float] = [0.8, 0.1, 0.1],
    ) -> None:
        """
        Add new data to the temporal splits.

        Args:
            new_data_path: Path to CSV file with new data
            split_ratios: Train/val/test split ratios (must sum to 1)
        """
        assert sum(split_ratios) == 1, "Split ratios must sum to 1"

        new_data = pd.read_csv(new_data_path)
        new_data["dyear"] = new_data["admissioninstant"].apply(lambda x: int(x[:4]))
        new_data["dmonth"] = new_data["admissioninstant"].apply(lambda x: int(x[5:7]))

        # Create train, val, and test splits within years
        unique_years = new_data["dyear"].unique()
        for year in unique_years:
            year_data = new_data[new_data["dyear"] == year]
            year_data = year_data.sample(frac=1)  # Shuffle

            train_size = int(len(year_data) * split_ratios[0])
            val_size = int(len(year_data) * split_ratios[1])

            train_data = year_data.iloc[:train_size]
            val_data = year_data.iloc[train_size : train_size + val_size]
            test_data = year_data.iloc[train_size + val_size :]

            train_data.to_parquet(self.data_dir / f"train_{year}.parquet", index=False)
            val_data.to_parquet(self.data_dir / f"val_{year}.parquet", index=False)
            test_data.to_parquet(self.data_dir / f"test_{year}.parquet", index=False)

    def prepare_data(self) -> None:
        """
        Prepare data by creating year-specific parquet files from CSVs.

        This method is called only once and creates the temporal splits.
        """
        # Check if CSV files exist
        test_csv = self.data_dir / "test.csv"
        train_csv = self.data_dir / "train.csv"
        val_csv = self.data_dir / "val.csv"

        if not all(f.exists() for f in [test_csv, train_csv, val_csv]):
            return  # Data already prepared or using pre-split parquet files

        test_data = pd.read_csv(test_csv)
        train_data = pd.read_csv(train_csv)
        val_data = pd.read_csv(val_csv)

        unique_train_years = sorted(train_data["dyear"].unique())
        unique_val_years = sorted(val_data["dyear"].unique())
        unique_test_years = sorted(test_data["dyear"].unique())

        # Find years present in all splits
        unique_years = sorted(
            set(unique_train_years) & set(unique_val_years) & set(unique_test_years)
        )

        def _parquet_ready(path: Path) -> bool:
            """Return True when a parquet file already exists and is non-empty."""
            try:
                return path.exists() and path.stat().st_size > 0
            except OSError:
                return False

        def _year_has_all_parquet(year: int) -> bool:
            return all(
                _parquet_ready(self.data_dir / f"{split}_{year}.parquet")
                for split in ("train", "val", "test")
            )

        if unique_years and all(_year_has_all_parquet(year) for year in unique_years):
            return  # Parquet shards already materialized

        for year in unique_years:
            test_path = self.data_dir / f"test_{year}.parquet"
            if not _parquet_ready(test_path):
                test_data[test_data["dyear"] == year].to_parquet(test_path)

            val_path = self.data_dir / f"val_{year}.parquet"
            if not _parquet_ready(val_path):
                val_data[val_data["dyear"] == year].to_parquet(val_path)

            train_path = self.data_dir / f"train_{year}.parquet"
            if not _parquet_ready(train_path):
                train_data[train_data["dyear"] == year].to_parquet(train_path)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up dataloaders for the specified stage.

        Args:
            stage: One of 'fit', 'validate', 'test', or None (all stages)
        """
        self.train_loaders = []
        self.train_loader_class_weights = []
        self.val_loaders = []
        self.test_loaders = []

        if stage in ("fit", None):
            self._setup_train_loaders()

        if stage in ("fit", "validate", None):
            self._setup_val_loaders()

        if stage in ("test", None):
            self._setup_test_loaders()

    def _setup_train_loaders(self) -> None:
        """Set up training dataloaders."""
        if self.train_sequentially:
            for year in self.train_years:
                train_data = pd.read_parquet(self.data_dir / f"train_{year}.parquet")
                train_dataset = ReadmissionDataset(train_data, self.tokenizer, self.max_len)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )
                self.train_loaders.append(train_loader)

                # Compute class weights
                proportions = train_data["readmitted_in_30_days"].value_counts(normalize=True)
                class_weights = torch.tensor(
                    [1.0 / proportions[0], 1.0 / proportions[1]], dtype=torch.float32
                )
                self.train_loader_class_weights.append(class_weights)
        else:
            # Pool all training years
            train_data = pd.concat([
                pd.read_parquet(self.data_dir / f"train_{year}.parquet")
                for year in self.train_years
            ])
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            train_dataset = ReadmissionDataset(train_data, self.tokenizer, self.max_len)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            self.train_loaders.append(train_loader)

            proportions = train_data["readmitted_in_30_days"].value_counts(normalize=True)
            class_weights = torch.tensor(
                [1.0 / proportions[0], 1.0 / proportions[1]], dtype=torch.float32
            )
            self.train_loader_class_weights.append(class_weights)

    def _setup_val_loaders(self) -> None:
        """Set up validation dataloaders."""
        if self.validate_sequentially:
            for year in self.val_years:
                val_data = pd.read_parquet(self.data_dir / f"val_{year}.parquet")
                val_dataset = ReadmissionDataset(val_data, self.tokenizer, self.max_len)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
                self.val_loaders.append(val_loader)
        else:
            val_data = pd.concat([
                pd.read_parquet(self.data_dir / f"val_{year}.parquet")
                for year in self.val_years
            ])
            val_data = val_data.sample(frac=1).reset_index(drop=True)
            val_dataset = ReadmissionDataset(val_data, self.tokenizer, self.max_len)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            self.val_loaders.append(val_loader)

    def _setup_test_loaders(self) -> None:
        """Set up test dataloaders with natural class distribution."""
        if self.test_sequentially:
            for year in self.test_years:
                test_data = pd.read_parquet(self.data_dir / f"test_{year}.parquet")
                test_dataset = ReadmissionDataset(test_data, self.tokenizer, self.max_len)
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                )
                self.test_loaders.append(test_loader)
        else:
            test_data = pd.concat([
                pd.read_parquet(self.data_dir / f"test_{year}.parquet")
                for year in self.test_years
            ])
            test_dataset = ReadmissionDataset(test_data, self.tokenizer, self.max_len)
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            self.test_loaders.append(test_loader)

    def train_dataloader(self) -> List[DataLoader]:
        """Return training dataloaders."""
        return self.train_loaders

    def val_dataloader(self) -> List[DataLoader]:
        """Return validation dataloaders."""
        return self.val_loaders

    def test_dataloader(self) -> List[DataLoader]:
        """Return test dataloaders."""
        return self.test_loaders
