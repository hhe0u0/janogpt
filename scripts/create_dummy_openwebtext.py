#!/usr/bin/env python3
"""
Create a small dummy OpenWebText dataset for testing.

This creates train.bin and val.bin with random tokens for quick testing.
For real training, use the actual OpenWebText dataset.
"""

import argparse
from pathlib import Path

import numpy as np


def create_dummy_data(output_dir, num_tokens=1_000_000, vocab_size=50257, seed=42):
    """
    Create dummy train.bin and val.bin files.

    Args:
        output_dir: Directory to save files
        num_tokens: Number of tokens to generate
        vocab_size: Vocabulary size (50257 for GPT2)
        seed: Random seed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating dummy OpenWebText data...")
    print(f"  Output dir:   {output_dir}")
    print(f"  Total tokens: {num_tokens:,}")
    print(f"  Vocab size:   {vocab_size:,}")

    # Set seed for reproducibility
    np.random.seed(seed)

    # Split: 90% train, 10% val
    train_tokens = int(num_tokens * 0.9)
    val_tokens = num_tokens - train_tokens

    # Create train.bin
    print(f"\nGenerating train.bin ({train_tokens:,} tokens)...")
    train_data = np.random.randint(0, vocab_size, size=train_tokens, dtype=np.uint16)
    train_file = output_dir / "train.bin"
    train_data.tofile(str(train_file))
    print(f"  Saved: {train_file} ({train_file.stat().st_size / 1e6:.1f} MB)")

    # Create val.bin
    print(f"\nGenerating val.bin ({val_tokens:,} tokens)...")
    val_data = np.random.randint(0, vocab_size, size=val_tokens, dtype=np.uint16)
    val_file = output_dir / "val.bin"
    val_data.tofile(str(val_file))
    print(f"  Saved: {val_file} ({val_file.stat().st_size / 1e6:.1f} MB)")

    # Create meta.pkl (optional metadata)
    print(f"\nCreating meta.pkl...")
    import pickle
    meta = {
        'vocab_size': vocab_size,
        'train_tokens': train_tokens,
        'val_tokens': val_tokens,
        'note': 'This is dummy data for testing. Use real OpenWebText for training.'
    }
    meta_file = output_dir / "meta.pkl"
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    print(f"  Saved: {meta_file}")

    print(f"\n{'='*60}")
    print(f"✓ Dummy data created successfully!")
    print(f"\nYou can now run:")
    print(f"  python scripts/eval_pretrained_loss.py --data_dir {output_dir}")
    print(f"\nNOTE: This is random data. Loss will be ~10.8 (theoretical max).")
    print(f"      For meaningful results, prepare real OpenWebText data.")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Create dummy OpenWebText dataset')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/openwebtext',
        help='Output directory'
    )
    parser.add_argument(
        '--num_tokens',
        type=int,
        default=1_000_000,
        help='Total number of tokens to generate (default: 1M)'
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=50257,
        help='Vocabulary size (default: 50257 for GPT2)'
    )

    args = parser.parse_args()

    create_dummy_data(
        output_dir=args.output_dir,
        num_tokens=args.num_tokens,
        vocab_size=args.vocab_size
    )


if __name__ == '__main__':
    main()
