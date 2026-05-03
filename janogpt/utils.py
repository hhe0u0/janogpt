"""
Utility functions and data loaders.
"""

import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Dict

import numpy as np

from janogpt.protocols import DataLoader as BaseDataLoader

# ========== Data Loaders ==========


class DummyDataLoader(BaseDataLoader):
    """Simple in-memory data loader for testing."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        vocab_size: int,
        num_batches: int = 1,
        seed: int = 42,
    ):
        """
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            vocab_size: Vocabulary size
            num_batches: Number of batches to generate
            seed: Random seed
        """
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_batches = num_batches
        self.rng = np.random.RandomState(seed)

        # Pre-generate all batches
        self.batches = []
        for _ in range(num_batches):
            batch = self.rng.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
            self.batches.append({"input_ids": batch})

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Yield pre-generated batches, then loop."""
        idx = 0
        while True:
            yield self.batches[idx % self.num_batches]
            idx += 1

    def get_batch(self) -> Dict[str, np.ndarray]:
        """Get first batch."""
        return self.batches[0]


class FileDataLoader(BaseDataLoader):
    """Memory-mapped file data loader for large datasets."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        seq_len: int,
        split: str = "train",
        seed: int = None,
    ):
        """
        Args:
            data_dir: Directory containing train.bin and val.bin
            batch_size: Global batch size
            seq_len: Sequence length
            split: 'train' or 'val'
            seed: Random seed
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split

        # Load memmap file
        data_file = self.data_dir / f"{split}.bin"
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        self.data = np.memmap(str(data_file), dtype=np.uint16, mode="r")
        print(f"Loaded {split} data: {len(self.data):,} tokens")

        # Setup random state
        self.rng = np.random.RandomState(seed)

        # Load metadata if available
        meta_file = self.data_dir / "meta.pkl"
        if meta_file.exists():
            with open(meta_file, "rb") as f:
                self.meta = pickle.load(f)
            print(f"Metadata: vocab_size={self.meta.get('vocab_size', 'unknown')}")
        else:
            self.meta = {}

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Infinite iterator yielding random batches."""
        while True:
            # Sample random starting positions
            max_start = len(self.data) - self.seq_len
            ix = self.rng.randint(0, max_start, size=(self.batch_size,))

            # Extract sequences
            x = np.stack([self.data[i : i + self.seq_len].astype(np.int32) for i in ix])

            yield {"input_ids": x}

    def get_batch(self) -> Dict[str, np.ndarray]:
        """Get a single batch."""
        return next(iter(self))


# ========== Batch Sharding Utilities ==========


def shard_batch(batch: Dict[str, np.ndarray], num_devices: int) -> Dict[str, np.ndarray]:
    """
    Shard batch across devices: (B, T) -> (num_devices, B//num_devices, T).
    """

    def _shard(x):
        assert x.shape[0] % num_devices == 0, (
            f"Batch size {x.shape[0]} not divisible by {num_devices} devices"
        )
        return x.reshape(num_devices, x.shape[0] // num_devices, *x.shape[1:])

    return {k: _shard(v) for k, v in batch.items()}


def shard_accum_batch(
    batch: Dict[str, np.ndarray],
    num_devices: int,
    accum_steps: int,
) -> Dict[str, np.ndarray]:
    """
    Shard batch for gradient accumulation.

    Input:  (global_batch, seq_len)
    Output: (num_devices, accum_steps, micro_batch, seq_len)
    """

    def _shard(x):
        B = x.shape[0]
        micro = B // (num_devices * accum_steps)
        assert num_devices * accum_steps * micro == B, (
            f"Batch size {B} not divisible by {num_devices} × {accum_steps}"
        )
        return x.reshape(num_devices, accum_steps, micro, *x.shape[1:])

    return {k: _shard(v) for k, v in batch.items()}
