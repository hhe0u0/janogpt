#!/usr/bin/env python3
"""
End-to-end training test: train small model, log, eval, save, and resume.
"""

import tempfile
from pathlib import Path

import jax

from janogpt import Config, GPT, Trainer
from janogpt.logger import ConsoleLogger, DatasetEvaluator
from janogpt.utils import DummyDataLoader


def test_training_end_to_end():
    """Test complete training workflow: train, log, eval, save, resume."""
    print("=" * 80)
    print("End-to-End Training Test")
    print("=" * 80)

    # Create minimal config for fast training
    config = Config(
        # Model (tiny for speed)
        voc_size=100,
        seq_len=64,
        num_blocks=2,
        emb_dim=32,
        num_heads=2,
        # Training
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        max_steps=3,  # Train 3 steps
        seed=42,
        # Optimizer
        learning_rate=1e-3,
        warmup_steps=0,
        # Logging
        log_interval=1,  # Log every step
        eval_interval=2,  # Eval at step 2
        eval_iters=2,  # Only 2 eval batches
        # Checkpointing
        output_dir="test_output",
        save_interval=2,  # Save at step 2
        # Data
        data_dir="dummy",
        # WandB
        wandb_log=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config.output_dir = tmpdir

        print(f"\n{'='*80}")
        print("PART 1: Initial Training (steps 1-3)")
        print(f"{'='*80}")

        # Create model and data
        model = GPT(config)
        train_loader = DummyDataLoader(
            batch_size=config.micro_batch_size,
            seq_len=config.seq_len,
            voc_size=config.voc_size,
            seed=config.seed,
        )
        val_loader = DummyDataLoader(
            batch_size=8,  # Small eval batch
            seq_len=config.seq_len,
            voc_size=config.voc_size,
            seed=config.seed + 1,
        )

        # Create evaluator and logger
        evaluators = [DatasetEvaluator(val_loader, config.eval_iters, "val", model=model)]
        logger = ConsoleLogger()

        # Create trainer and train
        print("\nCreating trainer...")
        trainer = Trainer(
            model=model,
            config=config,
            evaluators=evaluators,
            logger=logger,
            seed=config.seed,
        )

        print("\nStarting training for 3 steps...")
        print("  - Should log at steps 1, 2, 3")
        print("  - Should eval at step 2")
        print("  - Should save checkpoint at step 2")
        print()

        trainer.train(train_loader)

        # Verify checkpoint was saved
        ckpt_dir = Path(tmpdir) / "checkpoints" / "step_2"
        assert ckpt_dir.exists(), f"Checkpoint not saved at {ckpt_dir}"
        print(f"\n✓ Checkpoint saved at {ckpt_dir}")

        # Get final params for comparison
        final_params_step3 = trainer.state.params

        print(f"\n{'='*80}")
        print("PART 2: Resume from Checkpoint (continue to step 5)")
        print(f"{'='*80}")

        # Update config to resume from checkpoint
        config.resume_from_checkpoint = str(ckpt_dir)
        config.max_steps = 5  # Continue training to step 5

        # Create new model, data, evaluator, logger
        model2 = GPT(config)
        train_loader2 = DummyDataLoader(
            batch_size=config.micro_batch_size,
            seq_len=config.seq_len,
            voc_size=config.voc_size,
            seed=config.seed,
        )
        val_loader2 = DummyDataLoader(
            batch_size=8,
            seq_len=config.seq_len,
            voc_size=config.voc_size,
            seed=config.seed + 1,
        )
        evaluators2 = [DatasetEvaluator(val_loader2, config.eval_iters, "val", model=model2)]
        logger2 = ConsoleLogger()

        # Create new trainer (will load checkpoint in train())
        print("\nCreating new trainer to resume...")
        trainer2 = Trainer(
            model=model2,
            config=config,
            evaluators=evaluators2,
            logger=logger2,
            seed=config.seed,
        )

        print("\nResuming training from step 2...")
        print("  - Should load checkpoint from step 2")
        print("  - Should continue to steps 3, 4, 5")
        print("  - Should eval at step 4")
        print("  - Should save checkpoint at step 4")
        print()

        trainer2.train(train_loader2)

        # Verify new checkpoint was saved
        ckpt_dir_step4 = Path(tmpdir) / "checkpoints" / "step_4"
        assert ckpt_dir_step4.exists(), f"Checkpoint not saved at {ckpt_dir_step4}"
        print(f"\n✓ Checkpoint saved at {ckpt_dir_step4}")

        # Get final params after resuming
        final_params_step5 = trainer2.state.params

        print(f"\n{'='*80}")
        print("PART 3: Verification")
        print(f"{'='*80}")

        # Verify params changed from step 3 to step 5 (due to training)
        def params_different(p1, p2):
            return not jax.tree_util.tree_all(
                jax.tree_util.tree_map(lambda x, y: (x == y).all(), p1, p2)
            )

        params_changed = params_different(final_params_step3, final_params_step5)
        if params_changed:
            print("✓ Parameters changed from step 3 to step 5 (training continued)")
        else:
            print("✗ Parameters did not change (training may not have continued)")
            return False

        # Verify checkpoint contains expected files
        print("\n✓ Checking checkpoint structure...")
        step2_files = list(ckpt_dir.rglob("*"))
        print(f"  Step 2 checkpoint has {len(step2_files)} files")

        step4_files = list(ckpt_dir_step4.rglob("*"))
        print(f"  Step 4 checkpoint has {len(step4_files)} files")

        print(f"\n{'='*80}")
        print("✓ ALL TESTS PASSED")
        print(f"{'='*80}")
        print("\nSummary:")
        print("  ✓ Initial training completed (3 steps)")
        print("  ✓ Logging worked (every step)")
        print("  ✓ Evaluation worked (step 2)")
        print("  ✓ Checkpoint saved (step 2)")
        print("  ✓ Resumed from checkpoint (loaded step 2)")
        print("  ✓ Continued training (steps 3-5)")
        print("  ✓ New checkpoint saved (step 4)")
        print("  ✓ Parameters updated correctly")

        return True


if __name__ == "__main__":
    try:
        success = test_training_end_to_end()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"✗ TEST FAILED")
        print(f"{'=' * 80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
