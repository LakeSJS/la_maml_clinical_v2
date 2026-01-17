"""Shared pytest fixtures for LA-MAML tests."""

import pytest
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@pytest.fixture
def test_config():
    """Base test configuration."""
    return {
        "model_name": "distilbert-base-uncased",  # Smaller model for testing
        "batch_size": 2,
        "seq_len": 64,
        "num_labels": 2,
        "learning_rate": 1e-3,
        "inner_loop_learning_rate": 1e-3,
        "nu_lr": 1e-3,
        "alpha_0": 1e-2,
        "buffer_size": 5,
    }


@pytest.fixture
def sample_batch(test_config):
    """Create a sample batch for testing."""
    batch_size = test_config["batch_size"]
    seq_len = test_config["seq_len"]

    return {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len),
        "labels": torch.randint(0, 2, (batch_size,)),
    }


@pytest.fixture
def tokenizer(test_config):
    """Load tokenizer for testing."""
    return AutoTokenizer.from_pretrained(test_config["model_name"])


@pytest.fixture
def base_model(test_config):
    """Load base model for testing."""
    return AutoModelForSequenceClassification.from_pretrained(
        test_config["model_name"],
        num_labels=test_config["num_labels"],
    )
