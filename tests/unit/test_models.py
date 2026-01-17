"""Tests for model implementations."""

import pytest
import torch

from lamaml_clinical.models import (
    BaseModule,
    CmamlModule,
    LamamlModule,
    ReplayBuffer,
    TraditionalModule,
)


class TestReplayBuffer:
    """Test ReplayBuffer functionality."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = ReplayBuffer(buffer_size=10)
        assert len(buffer) == 0
        assert buffer.buffer_size == 10

    def test_expose_to_sample(self):
        """Test adding samples to buffer."""
        buffer = ReplayBuffer(buffer_size=5)
        sample = {
            "input_ids": torch.randint(0, 100, (1, 10)),
            "attention_mask": torch.ones(1, 10),
            "labels": torch.tensor([0]),
        }
        buffer.expose_to_sample(sample)
        assert len(buffer) == 1

    def test_reservoir_sampling(self):
        """Test that buffer doesn't exceed capacity."""
        buffer = ReplayBuffer(buffer_size=5)
        for i in range(20):
            sample = {
                "input_ids": torch.randint(0, 100, (1, 10)),
                "attention_mask": torch.ones(1, 10),
                "labels": torch.tensor([i % 2]),
            }
            buffer.expose_to_sample(sample)
        assert len(buffer) == 5
        assert buffer.global_samples_seen == 20

    def test_get_samples(self):
        """Test retrieving samples from buffer."""
        buffer = ReplayBuffer(buffer_size=10)
        for i in range(10):
            sample = {
                "input_ids": torch.randint(0, 100, (1, 10)),
                "attention_mask": torch.ones(1, 10),
                "labels": torch.tensor([i % 2]),
            }
            buffer.expose_to_sample(sample)

        samples = buffer.get_samples(5)
        assert "input_ids" in samples
        assert samples["input_ids"].shape[0] == 5

    def test_get_samples_empty_buffer(self):
        """Test getting samples from empty buffer."""
        buffer = ReplayBuffer(buffer_size=10)
        samples = buffer.get_samples(5)
        assert samples == {}

    def test_local_sample_limit(self):
        """Test local sample limit functionality."""
        buffer = ReplayBuffer(buffer_size=10, local_sample_limit=3)
        for i in range(10):
            sample = {
                "input_ids": torch.randint(0, 100, (1, 10)),
                "attention_mask": torch.ones(1, 10),
                "labels": torch.tensor([i % 2]),
            }
            buffer.expose_to_sample(sample)
        assert len(buffer) == 3  # Limited by local_sample_limit


class TestTraditionalModule:
    """Test TraditionalModule."""

    def test_initialization(self, base_model, tokenizer, test_config):
        """Test module initialization."""
        module = TraditionalModule(
            model=base_model,
            tokenizer=tokenizer,
            learning_rate=test_config["learning_rate"],
            buffer_size=test_config["buffer_size"],
        )
        assert module.learning_rate == test_config["learning_rate"]
        assert len(module.replay_buffer) == 0

    def test_configure_optimizers(self, base_model, tokenizer, test_config):
        """Test optimizer configuration."""
        module = TraditionalModule(
            model=base_model,
            tokenizer=tokenizer,
            learning_rate=test_config["learning_rate"],
        )
        optimizer = module.configure_optimizers()
        assert isinstance(optimizer, torch.optim.AdamW)


class TestCmamlModule:
    """Test CmamlModule."""

    def test_initialization(self, base_model, tokenizer, test_config):
        """Test CMAML module initialization."""
        module = CmamlModule(
            model=base_model,
            tokenizer=tokenizer,
            inner_loop_learning_rate=test_config["inner_loop_learning_rate"],
            learning_rate=test_config["learning_rate"],
            buffer_size=test_config["buffer_size"],
        )
        assert module.inner_loop_learning_rate == test_config["inner_loop_learning_rate"]
        assert module.automatic_optimization is False

    def test_configure_optimizers(self, base_model, tokenizer, test_config):
        """Test optimizer configuration."""
        module = CmamlModule(
            model=base_model,
            tokenizer=tokenizer,
            inner_loop_learning_rate=test_config["inner_loop_learning_rate"],
            learning_rate=test_config["learning_rate"],
        )
        optimizers = module.configure_optimizers()
        assert isinstance(optimizers, list)
        assert len(optimizers) == 1


class TestLamamlModule:
    """Test LamamlModule."""

    def test_initialization(self, base_model, tokenizer, test_config):
        """Test LA-MAML module initialization."""
        module = LamamlModule(
            model=base_model,
            tokenizer=tokenizer,
            nu_lr=test_config["nu_lr"],
            alpha_0=test_config["alpha_0"],
            buffer_size=test_config["buffer_size"],
        )
        assert module.nu_lr == test_config["nu_lr"]
        assert module.alpha_0 == test_config["alpha_0"]
        assert module.automatic_optimization is False

    def test_learnable_lrs_initialized(self, base_model, tokenizer, test_config):
        """Test that learnable learning rates are initialized."""
        module = LamamlModule(
            model=base_model,
            tokenizer=tokenizer,
            nu_lr=test_config["nu_lr"],
            alpha_0=test_config["alpha_0"],
        )
        assert len(module.learnable_lrs) > 0
        for lr in module.learnable_lrs:
            assert torch.allclose(lr, torch.tensor(test_config["alpha_0"]))

    def test_configure_optimizers(self, base_model, tokenizer, test_config):
        """Test optimizer configuration returns two optimizers."""
        module = LamamlModule(
            model=base_model,
            tokenizer=tokenizer,
            nu_lr=test_config["nu_lr"],
            alpha_0=test_config["alpha_0"],
        )
        optimizers = module.configure_optimizers()
        assert isinstance(optimizers, list)
        assert len(optimizers) == 2  # meta_optimizer and lr_optimizer


class TestBaseModule:
    """Test BaseModule functionality."""

    def test_build_meta_batch_empty_buffer(self, base_model, tokenizer, sample_batch):
        """Test meta-batch with empty buffer returns original batch."""
        module = TraditionalModule(
            model=base_model,
            tokenizer=tokenizer,
        )
        meta_batch = module.build_meta_batch(sample_batch)
        assert meta_batch["input_ids"].shape == sample_batch["input_ids"].shape

    def test_build_meta_batch_with_buffer(self, base_model, tokenizer, sample_batch):
        """Test meta-batch combines current batch with buffer samples."""
        module = TraditionalModule(
            model=base_model,
            tokenizer=tokenizer,
            buffer_size=10,
        )
        # Add samples to buffer
        for i in range(5):
            module.replay_buffer.expose_to_sample({
                "input_ids": torch.randint(0, 100, (1, sample_batch["input_ids"].shape[1])),
                "attention_mask": torch.ones(1, sample_batch["input_ids"].shape[1]),
                "labels": torch.tensor([i % 2]),
            })

        meta_batch = module.build_meta_batch(sample_batch)
        # Meta-batch should be larger than original
        assert meta_batch["input_ids"].shape[0] > sample_batch["input_ids"].shape[0]
