"""Unit tests for utilities."""

import jax.numpy as jnp
import numpy as np
import pytest

from janogpt.utils import DummyDataLoader, FileDataLoader, shard_accum_batch, shard_batch


class TestDummyDataLoader:
    """Test DummyDataLoader."""

    def test_dummy_dataloader_init(self):
        """Test DummyDataLoader initializes correctly."""
        loader = DummyDataLoader(
            batch_size=4,
            seq_len=16,
            voc_size=256,
        )

        assert loader.batch_size == 4
        assert loader.seq_len == 16
        assert loader.voc_size == 256

    def test_dummy_dataloader_yields_correct_shape(self):
        """Test DummyDataLoader yields correct batch shape."""
        loader = DummyDataLoader(
            batch_size=4,
            seq_len=16,
            voc_size=256,
        )

        batch = next(iter(loader))

        assert "input_ids" in batch
        assert batch["input_ids"].shape == (4, 16)

    def test_dummy_dataloader_tokens_in_range(self):
        """Test DummyDataLoader tokens are within vocab range."""
        loader = DummyDataLoader(
            batch_size=4,
            seq_len=16,
            voc_size=256,
        )

        batch = next(iter(loader))
        tokens = batch["input_ids"]

        assert np.all(tokens >= 0)
        assert np.all(tokens < 256)

    def test_dummy_dataloader_is_deterministic(self):
        """Test DummyDataLoader is deterministic (always same batch)."""
        loader = DummyDataLoader(
            batch_size=4,
            seq_len=16,
            voc_size=256,
        )

        batch1 = next(iter(loader))
        batch2 = next(iter(loader))

        # Should be identical
        assert np.array_equal(batch1["input_ids"], batch2["input_ids"])


class TestFileDataLoader:
    """Test FileDataLoader."""

    def test_file_dataloader_init_missing_file(self):
        """Test FileDataLoader raises error for missing file."""
        with pytest.raises(FileNotFoundError):
            FileDataLoader(
                data_dir="nonexistent_dir",
                batch_size=4,
                seq_len=16,
                split="train",
            )

    def test_file_dataloader_with_dummy_file(self, tmp_path):
        """Test FileDataLoader with temporary data file."""
        # Create dummy data file
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        train_file = data_dir / "train.bin"
        data = np.arange(10000, dtype=np.uint16)
        data.tofile(str(train_file))

        # Create loader
        loader = FileDataLoader(
            data_dir=str(data_dir),
            batch_size=4,
            seq_len=16,
            split="train",
            seed=42,
        )

        # Get batch
        batch = next(iter(loader))

        assert "input_ids" in batch
        assert batch["input_ids"].shape == (4, 16)
        assert batch["input_ids"].dtype == np.int32

    def test_file_dataloader_infinite_iterator(self, tmp_path):
        """Test FileDataLoader yields infinite batches."""
        # Create dummy data
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        train_file = data_dir / "train.bin"
        data = np.arange(10000, dtype=np.uint16)
        data.tofile(str(train_file))

        loader = FileDataLoader(
            data_dir=str(data_dir),
            batch_size=4,
            seq_len=16,
            split="train",
            seed=42,
        )

        # Should be able to get many batches
        batches = [next(iter(loader)) for _ in range(10)]
        assert len(batches) == 10

    def test_file_dataloader_random_sampling(self, tmp_path):
        """Test FileDataLoader samples randomly."""
        # Create dummy data
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        train_file = data_dir / "train.bin"
        data = np.arange(10000, dtype=np.uint16)
        data.tofile(str(train_file))

        loader = FileDataLoader(
            data_dir=str(data_dir),
            batch_size=4,
            seq_len=16,
            split="train",
            seed=42,
        )

        # Get two batches
        batch1 = next(iter(loader))
        batch2 = next(iter(loader))

        # Should be different (random sampling)
        assert not np.array_equal(batch1["input_ids"], batch2["input_ids"])


class TestBatchSharding:
    """Test batch sharding utilities."""

    def test_shard_batch_single_device(self):
        """Test sharding for single device."""
        batch = {"input_ids": np.arange(16).reshape(4, 4)}

        sharded = shard_batch(batch, num_devices=1)

        # (1, 4, 4) - one device, 4 seqs, 4 tokens
        assert sharded["input_ids"].shape == (1, 4, 4)

    def test_shard_batch_multi_device(self):
        """Test sharding for multiple devices."""
        batch = {"input_ids": np.arange(32).reshape(8, 4)}

        sharded = shard_batch(batch, num_devices=2)

        # (2, 4, 4) - two devices, 4 seqs each, 4 tokens
        assert sharded["input_ids"].shape == (2, 4, 4)

    def test_shard_batch_not_divisible(self):
        """Test error when batch size not divisible by num_devices."""
        batch = {"input_ids": np.arange(15).reshape(5, 3)}

        with pytest.raises(AssertionError):
            shard_batch(batch, num_devices=2)

    def test_shard_accum_batch(self):
        """Test gradient accumulation sharding."""
        batch = {"input_ids": np.arange(64).reshape(16, 4)}

        sharded = shard_accum_batch(
            batch,
            num_devices=2,
            accum_steps=4,
        )

        # (2, 4, 2, 4) - 2 devices, 4 accum steps, 2 micro batch, 4 tokens
        assert sharded["input_ids"].shape == (2, 4, 2, 4)

    def test_shard_accum_batch_calculation(self):
        """Test gradient accumulation sharding math."""
        # 32 sequences total
        # 4 devices × 2 accum steps × 4 micro batch = 32
        batch = {"input_ids": np.arange(32 * 8).reshape(32, 8)}

        sharded = shard_accum_batch(
            batch,
            num_devices=4,
            accum_steps=2,
        )

        # (4, 2, 4, 8)
        assert sharded["input_ids"].shape == (4, 2, 4, 8)

    def test_shard_accum_batch_not_divisible(self):
        """Test error when batch size not divisible."""
        batch = {"input_ids": np.arange(30).reshape(10, 3)}

        with pytest.raises(AssertionError):
            shard_accum_batch(
                batch,
                num_devices=2,
                accum_steps=3,
            )

    def test_shard_preserves_data(self):
        """Test sharding preserves all data."""
        original = np.arange(16).reshape(8, 2)
        batch = {"input_ids": original}

        sharded = shard_batch(batch, num_devices=2)

        # Flatten and check all values present
        flattened = sharded["input_ids"].reshape(-1)
        assert np.array_equal(np.sort(flattened), np.arange(16))


class TestBatchShardingWithJAX:
    """Test batch sharding with JAX arrays."""

    def test_shard_with_jax_arrays(self):
        """Test sharding works with JAX arrays."""
        batch = {"input_ids": jnp.arange(16).reshape(4, 4)}

        sharded = shard_batch(batch, num_devices=2)

        assert sharded["input_ids"].shape == (2, 2, 4)
        assert isinstance(sharded["input_ids"], jnp.ndarray)

    def test_shard_accum_with_jax_arrays(self):
        """Test accumulation sharding with JAX arrays."""
        batch = {"input_ids": jnp.arange(32).reshape(8, 4)}

        sharded = shard_accum_batch(
            batch,
            num_devices=2,
            accum_steps=2,
        )

        assert sharded["input_ids"].shape == (2, 2, 2, 4)
        assert isinstance(sharded["input_ids"], jnp.ndarray)
