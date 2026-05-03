#!/usr/bin/env python3
"""
Evaluate pretrained GPT2 models on OpenWebText data.

This establishes the target loss - what we should achieve when training from scratch.
The untrained model has loss ~10.8, pretrained models should be much lower (2-4 range).
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from janogpt import Config, GPT
from janogpt.utils import FileDataLoader
from pretrained.huggingface.loader import load_hf_gpt2_weights


def compute_loss_on_batch(model, params, batch):
    """
    Compute loss for a batch (no gradients).

    Args:
        model: GPT model
        params: Model parameters
        batch: Dict with 'input_ids'

    Returns:
        Loss (scalar)
    """
    input_ids = jnp.array(batch['input_ids'])

    # Forward pass (inference mode)
    logits = model.apply(
        {'params': params},
        input_ids,
        inference=True
    )

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    # Cross-entropy loss
    import optax
    loss = optax.softmax_cross_entropy_with_integer_labels(
        shift_logits, shift_labels
    ).mean()

    return loss


def evaluate_model(model_name, data_dir, num_batches=10, batch_size=4, seq_len=1024):
    """
    Evaluate a pretrained model on OpenWebText data.

    Args:
        model_name: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
        data_dir: Path to data directory with train.bin
        num_batches: Number of batches to evaluate
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Average loss
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")

    # Create config for model
    if model_name == 'gpt2':
        config = Config(
            num_blocks=12,
            emb_dim=768,
            num_heads=12,
            seq_len=seq_len,
            voc_size=50257,  # HF vocab size
        )
    elif model_name == 'gpt2-medium':
        config = Config(
            num_blocks=24,
            emb_dim=1024,
            ff_dim=4096,  # 4 * 1024
            num_heads=16,
            seq_len=seq_len,
            voc_size=50257,
        )
    elif model_name == 'gpt2-large':
        config = Config(
            num_blocks=36,
            emb_dim=1280,
            num_heads=20,
            seq_len=seq_len,
            voc_size=50257,
        )
    elif model_name == 'gpt2-xl':
        config = Config(
            num_blocks=48,
            emb_dim=1600,
            num_heads=25,
            seq_len=seq_len,
            voc_size=50257,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Config: {config.num_blocks} blocks, {config.emb_dim} dim, {config.num_heads} heads")

    # Create model
    print("Creating model...")
    model = GPT(config)

    # Load pretrained weights
    print(f"Loading pretrained weights from HuggingFace...")
    params = load_hf_gpt2_weights(model, config, model_name=model_name)

    # Count parameters
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Parameters: {param_count / 1e6:.1f}M")

    # Create data loader
    print(f"\nLoading data from {data_dir}...")
    try:
        loader = FileDataLoader(
            data_dir=data_dir,
            batch_size=batch_size,
            seq_len=seq_len,
            split='train',
            seed=42
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"\nMake sure you have prepared the data:")
        print(f"  1. Download: python scripts/download_openwebtext.py")
        print(f"  2. Prepare: python data/openwebtext/prepare.py")
        return None

    # Evaluate on num_batches
    print(f"\nEvaluating on {num_batches} batches...")
    losses = []

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        loss = compute_loss_on_batch(model, params, batch)
        losses.append(float(loss))

        print(f"  Batch {i+1}/{num_batches}: loss = {loss:.4f}")

    avg_loss = np.mean(losses)
    std_loss = np.std(losses)
    perplexity = np.exp(avg_loss)

    print(f"\n{'='*60}")
    print(f"Results for {model_name}:")
    print(f"  Average Loss: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"  Perplexity:   {perplexity:.2f}")
    print(f"{'='*60}")

    return avg_loss, std_loss, perplexity


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pretrained GPT2 models on OpenWebText'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/openwebtext',
        help='Path to data directory with train.bin'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['gpt2', 'gpt2-medium'],
        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
        help='Models to evaluate'
    )
    parser.add_argument(
        '--num_batches',
        type=int,
        default=10,
        help='Number of batches to evaluate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=1024,
        help='Sequence length'
    )

    args = parser.parse_args()

    # Check if data exists
    data_path = Path(args.data_dir) / 'train.bin'
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print(f"\nPlease prepare the data first:")
        print(f"  1. Download OpenWebText (if not already done)")
        print(f"  2. cd data/openwebtext")
        print(f"  3. python prepare.py")
        return

    print(f"\n{'#'*60}")
    print(f"# Pretrained Model Loss Evaluation")
    print(f"#")
    print(f"# Purpose: Establish target loss for training from scratch")
    print(f"# Baseline: Untrained model has loss ~10.8")
    print(f"# Target:   Pretrained models should be in 2-4 range")
    print(f"{'#'*60}")

    print(f"\nConfiguration:")
    print(f"  Data:        {args.data_dir}")
    print(f"  Models:      {', '.join(args.models)}")
    print(f"  Batches:     {args.num_batches}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Seq length:  {args.seq_len}")
    print(f"  Total tokens: {args.num_batches * args.batch_size * args.seq_len:,}")

    # Evaluate each model
    results = {}
    for model_name in args.models:
        try:
            result = evaluate_model(
                model_name,
                args.data_dir,
                num_batches=args.num_batches,
                batch_size=args.batch_size,
                seq_len=args.seq_len
            )
            if result is not None:
                results[model_name] = result
        except Exception as e:
            print(f"\nERROR evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if results:
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<15} {'Loss':<12} {'Perplexity':<12} {'vs Untrained'}")
        print(f"{'-'*60}")

        untrained_loss = 10.8
        for model_name, (loss, std, ppl) in results.items():
            improvement = untrained_loss - loss
            improvement_pct = (improvement / untrained_loss) * 100
            print(f"{model_name:<15} {loss:>6.4f} ± {std:.4f}  {ppl:>8.2f}      "
                  f"-{improvement:.2f} ({improvement_pct:.1f}% better)")

        print(f"\n{'='*60}")
        print(f"TARGET LOSS FOR TRAINING FROM SCRATCH:")
        print(f"  Start:  ~10.8 (random initialization)")
        print(f"  Target: ~{min(r[0] for r in results.values()):.2f} "
              f"({min(results.keys())} performance)")
        print(f"{'='*60}\n")
    else:
        print("\nNo results to display. Check errors above.")


if __name__ == '__main__':
    main()
