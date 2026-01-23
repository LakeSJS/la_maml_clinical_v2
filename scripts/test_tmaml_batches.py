"""Test script for TMAML batch pairing system."""

import sys
sys.path.insert(0, "/gpfs/data/oermannlab/users/slj9342/la_maml_clinical_v2/src")

from pytorch_lightning.utilities import CombinedLoader
from lamaml_clinical.core.config import load_config
from lamaml_clinical.data.datamodules import TemporalDataModule
from transformers import BertTokenizer


def test_tmaml_batch_pairing():
    """Verify TMAML gets paired current/future batches."""
    
    # Load config
    config = load_config("tmaml_seq_2013_2024", paths_config="gpfs")
    tokenizer = BertTokenizer.from_pretrained(config.paths.tokenizer_path)
    
    # Simulate what runner does for year 2013 (current) + 2014 (future)
    current_year = 2013
    future_year = 2014
    
    dm_current = TemporalDataModule(
        data_dir=config.paths.data_dir,
        train_years=[current_year],
        val_years=config.data.validation_years,
        test_years=config.data.test_years,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_len=config.data.max_len,
        train_sequentially=True,
    )
    
    dm_future = TemporalDataModule(
        data_dir=config.paths.data_dir,
        train_years=[future_year],
        val_years=config.data.validation_years,
        test_years=config.data.test_years,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_len=config.data.max_len,
        train_sequentially=True,
    )
    
    dm_current.setup("fit")
    dm_future.setup("fit")
    
    current_loader = dm_current.train_dataloader()[0]
    future_loader = dm_future.train_dataloader()[0]
    
    combined = CombinedLoader(
        {"current": current_loader, "future": future_loader},
        mode="min_size",
    )
    
    # Test 1: Check batch structure
    print("=" * 60)
    print("TEST 1: Batch structure")
    print("=" * 60)
    
    batch = next(iter(combined))
    
    assert isinstance(batch, dict), f"Expected dict, got {type(batch)}"
    assert "current" in batch, "Missing 'current' key"
    assert "future" in batch, "Missing 'future' key"
    print("✓ Batch is dict with 'current' and 'future' keys")
    
    # Test 2: Check batch contents
    print("\n" + "=" * 60)
    print("TEST 2: Batch contents")
    print("=" * 60)
    
    for name in ["current", "future"]:
        b = batch[name]
        assert "input_ids" in b, f"Missing 'input_ids' in {name}"
        assert "attention_mask" in b, f"Missing 'attention_mask' in {name}"
        assert "labels" in b, f"Missing 'labels' in {name}"
        print(f"✓ {name}: input_ids={b['input_ids'].shape}, labels={b['labels'].shape}")
    
    # Test 3: Batches are independent (different data)
    print("\n" + "=" * 60)
    print("TEST 3: Batches are independent")
    print("=" * 60)
    
    current_ids = batch["current"]["input_ids"]
    future_ids = batch["future"]["input_ids"]
    
    # They should NOT be identical (different years)
    are_same = (current_ids == future_ids).all().item()
    assert not are_same, "Current and future batches are identical!"
    print("✓ Current and future batches contain different data")
    
    # Test 4: Multiple iterations work
    print("\n" + "=" * 60)
    print("TEST 4: Multiple iterations")
    print("=" * 60)
    
    count = 0
    for batch in combined:
        count += 1
        if count >= 5:
            break
    print(f"✓ Successfully iterated {count} batches")
    
    # Test 5: Simulate TMAML training_step unpacking
    print("\n" + "=" * 60)
    print("TEST 5: TMAML training_step unpacking")
    print("=" * 60)
    
    batch = next(iter(combined))
    
    # This is what TMAML.training_step does:
    if isinstance(batch, dict) and "current" in batch and "future" in batch:
        current_batch = batch["current"]
        future_batch = batch["future"]
        print("✓ Unpacking works as expected")
        print(f"  Current batch size: {current_batch['input_ids'].shape[0]}")
        print(f"  Future batch size: {future_batch['input_ids'].shape[0]}")
    else:
        raise AssertionError("Batch format doesn't match TMAML expectations!")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_tmaml_batch_pairing()
