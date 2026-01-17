"""Replay buffer implementation with reservoir sampling."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


def _to_cpu(sample_dict: Dict[str, torch.Tensor], pin: bool = False) -> Dict[str, torch.Tensor]:
    """Move sample dictionary to CPU."""
    return {
        k: v.detach().cpu().pin_memory() if pin else v.detach().cpu()
        for k, v in sample_dict.items()
    }


def _to_device(
    sample_dict: Dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool = True,
) -> Dict[str, torch.Tensor]:
    """Move sample dictionary to specified device."""
    return {k: v.to(device, non_blocking=non_blocking) for k, v in sample_dict.items()}


class ReplayBuffer:
    """
    Replay buffer with reservoir sampling for continual learning.

    Maintains a fixed-size buffer of past samples, using reservoir sampling
    to ensure each sample has an equal probability of being retained.

    Args:
        buffer_size: Maximum number of samples to store
        local_sample_limit: Optional limit on samples to consider per training phase
    """

    def __init__(self, buffer_size: int, local_sample_limit: Optional[int] = None):
        self.buffer_size = buffer_size
        self.local_sample_limit = local_sample_limit
        self.data: List[Dict[str, torch.Tensor]] = []
        self.global_samples_seen = 0
        self.local_samples_seen = 0

    def expose_to_sample(self, sample: Dict[str, torch.Tensor]) -> None:
        """
        Expose the buffer to a new sample, potentially adding it.

        Uses reservoir sampling to maintain uniform probability of retention.

        Args:
            sample: Dictionary with input_ids, attention_mask, and labels
        """
        sample = _to_cpu(sample)
        self.local_samples_seen += 1

        # Check local sample limit
        if (
            self.local_sample_limit is not None
            and self.local_samples_seen > self.local_sample_limit
        ):
            return

        self.global_samples_seen += 1

        if len(self.data) < self.buffer_size:
            # Buffer not full - add sample directly
            self.data.append({
                "input_ids": sample["input_ids"].to(torch.int64),
                "attention_mask": sample["attention_mask"],
                "labels": sample["labels"],
            })
        else:
            # Reservoir sampling: replace with probability buffer_size/n
            idx = np.random.randint(0, self.global_samples_seen)
            if idx < self.buffer_size:
                self.data[idx] = {
                    "input_ids": sample["input_ids"].to(torch.int64),
                    "attention_mask": sample["attention_mask"],
                    "labels": sample["labels"],
                }

    def sample_stream(self, stream) -> None:
        """Sample a stream of samples into the buffer."""
        for sample in stream:
            self.expose_to_sample(sample)

    def get_samples(
        self, num_samples: int, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get random samples from the buffer.

        Args:
            num_samples: Number of samples to retrieve
            device: Device to move samples to

        Returns:
            Dictionary with batched samples, or empty dict if buffer is empty
        """
        if not self.data:
            return {}

        chosen_samples = random.sample(self.data, min(num_samples, len(self.data)))
        collated_samples = default_collate(chosen_samples)
        collated_samples = {k: v.squeeze(1) for k, v in collated_samples.items()}

        if device is not None:
            return _to_device(collated_samples, device=device)
        return collated_samples

    def reset_local_counter(self) -> None:
        """Reset the local samples counter (call at start of each training phase)."""
        self.local_samples_seen = 0

    def __len__(self) -> int:
        """Return current number of samples in buffer."""
        return len(self.data)
