#!/usr/bin/env python3
"""
Training script for JanoGPT.

Usage:
    python scripts/train.py --config configs/smoke_test.json
    python scripts/train.py --config configs/train_1gpu_a100.json
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
from janogpt import GPT, Config, Trainer
from janogpt.utils import DummyDataLoader, FileDataLoader
from janogpt.logger import DatasetEvaluator, ConsoleLogger, WandBLogger, MultiLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Train JanoGPT')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config JSON file')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = Config.from_json(args.config)
    if args.no_wandb:
        config.wandb_log = False

    print("=" * 80)
    print("JanoGPT Training")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Devices: {jax.local_device_count()} {jax.devices()[0].platform}")
    print("=" * 80)

    # Calculate effective batch
    num_devices = jax.local_device_count()
    effective_batch = (
        config.micro_batch_size
        * config.gradient_accumulation_steps
        * num_devices
    )

    # Create data loaders
    if config.dataloader_class == "DummyDataLoader":
        print("Using DummyDataLoader")
        train_loader = DummyDataLoader(
            batch_size=effective_batch,
            seq_len=config.seq_len,
            vocab_size=config.voc_size,
            num_batches=1,
            seed=config.seed,
        )
        val_loader = DummyDataLoader(
            batch_size=effective_batch,
            seq_len=config.seq_len,
            vocab_size=config.voc_size,
            num_batches=1,
            seed=config.seed + 1,
        )
    else:
        print(f"Loading data from {config.data_dir}")
        train_loader = FileDataLoader(
            data_dir=config.data_dir,
            batch_size=effective_batch,
            seq_len=config.seq_len,
            split='train',
            seed=config.seed,
        )
        val_loader = FileDataLoader(
            data_dir=config.data_dir,
            batch_size=effective_batch,
            seq_len=config.seq_len,
            split='val',
            seed=config.seed + 1,
        )

    # Create model
    model = GPT(config)

    # Create evaluators
    evaluators = [DatasetEvaluator(val_loader, config.eval_iters, "val")]

    # Create loggers
    console = ConsoleLogger()
    wandb = WandBLogger(enabled=config.wandb_log)
    logger = MultiLogger([console, wandb])

    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        evaluators=evaluators,
        logger=logger,
        seed=config.seed
    )

    # Train
    try:
        trainer.train(train_loader)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        trainer.save_checkpoint(int(trainer.state.step))

    print("\n✓ Training complete!")


if __name__ == '__main__':
    main()
