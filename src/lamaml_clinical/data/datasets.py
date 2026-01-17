"""Dataset implementations for clinical NLP tasks."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import Dataset


class ReadmissionDataset(Dataset):
    """
    Dataset for 30-day hospital readmission prediction.

    Tokenizes clinical discharge notes and provides binary readmission labels.

    Args:
        data: DataFrame with 'text' and 'readmitted_in_30_days' columns
        tokenizer: HuggingFace tokenizer
        max_len: Maximum sequence length for tokenization
    """

    def __init__(self, data: pd.DataFrame, tokenizer: Any, max_len: int = 512):
        self.data = data["text"].tolist()
        self.readmission = data["readmitted_in_30_days"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with input_ids, attention_mask, and labels
        """
        text = self.data[idx]
        labels = torch.tensor(self.readmission[idx], dtype=torch.long)
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        inputs["input_ids"] = inputs["input_ids"].squeeze(0)
        inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)

        assert labels.dtype == torch.long, f"Expected labels to be torch.long, got {labels.dtype}"
        assert set(labels.unique().tolist()).issubset({0, 1}), (
            f"Expected binary labels, got {labels.unique().tolist()}"
        )

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }
