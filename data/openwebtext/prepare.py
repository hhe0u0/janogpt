#!/usr/bin/env python3
"""
Download and prepare OpenWebText dataset for JanoGPT training.

This script downloads the pre-tokenized OpenWebText GPT-2 dataset from Kaggle.

Usage:
    python data/openwebtext/prepare.py
    python data/openwebtext/prepare.py --output_dir data/openwebtext
"""

import argparse
import os
import shutil
from pathlib import Path


def download_dataset(output_dir: str, force: bool = False):
    """Download OpenWebText dataset from Kaggle."""
    try:
        import kagglehub
    except ImportError:
        print("Error: kagglehub not installed")
        print("Run: pip install kagglehub")
        return False

    output_path = Path(output_dir)
    train_bin = output_path / "train.bin"
    val_bin = output_path / "val.bin"

    # Check if already downloaded
    if train_bin.exists() and val_bin.exists() and not force:
        print(f"✓ Dataset already exists at {output_dir}")
        print(f"  train.bin: {train_bin.stat().st_size / 1e9:.2f} GB")
        print(f"  val.bin: {val_bin.stat().st_size / 1e9:.2f} GB")
        return True

    print("=" * 80)
    print("Downloading OpenWebText GPT-2 Dataset")
    print("=" * 80)
    print("Dataset: windmaple/openwebtext-gpt2")
    print(f"Output: {output_dir}")
    print("=" * 80)
    print("\nDownloading... (this may take several minutes)\n")

    # Download from Kaggle
    dataset_path = kagglehub.dataset_download("windmaple/openwebtext-gpt2")
    print(f"✓ Downloaded to Kaggle cache: {dataset_path}")

    # Find the actual data files (they may be nested)
    dataset_path = Path(dataset_path)
    train_src = None
    val_src = None

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file == "train.bin":
                train_src = Path(root) / file
            elif file == "val.bin":
                val_src = Path(root) / file

    if not train_src or not val_src:
        print("Error: Could not find train.bin or val.bin in downloaded dataset")
        return False

    # Copy to output directory
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"\nCopying files to {output_dir}...")
    shutil.copy2(train_src, train_bin)
    shutil.copy2(val_src, val_bin)

    print("\n✓ Dataset ready!")
    print(f"  train.bin: {train_bin.stat().st_size / 1e9:.2f} GB")
    print(f"  val.bin: {val_bin.stat().st_size / 1e9:.2f} GB")

    return True


def main():
    parser = argparse.ArgumentParser(description="Download OpenWebText dataset")
    parser.add_argument(
        "--output_dir", type=str, default="data/openwebtext", help="Output directory for dataset"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download even if files exist"
    )
    args = parser.parse_args()

    success = download_dataset(args.output_dir, args.force)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
