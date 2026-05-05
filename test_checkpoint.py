#!/usr/bin/env python3
"""
Test checkpoint save/load functionality.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp

from janogpt import Config, GPT, Trainer
from janogpt.utils import DummyDataLoader


def test_checkpoint_save_load():
    """Test that checkpoints can be saved and loaded correctly."""
    print("=" * 80)
    print("Testing Checkpoint Save/Load")
    print("=" * 80)

    # Create minimal config
    config = Config(
        # Model
        voc_size=100,
        seq_len=128,
        num_blocks=2,
        emb_dim=64,
        num_heads=2,
        # Training
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=10,
        seed=42,
        # Optimizer
        learning_rate=1e-3,
        warmup_steps=0,
        # Logging
        log_interval=5,
        eval_interval=10,
        # Checkpointing
        output_dir="test_output",
        save_interval=5,
        # Data
        data_dir="dummy",
        # WandB
        wandb_log=False,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        config.output_dir = tmpdir

        print(f"\n1. Creating trainer with config...")
        print(f"   Output dir: {config.output_dir}")

        # Create trainer
        model = GPT(config)
        trainer = Trainer(model=model, config=config, evaluators=None, logger=None, seed=42)

        # Get initial state values
        initial_params = trainer.state.params
        initial_step = int(trainer.state.step)
        print(f"   Initial step: {initial_step}")
        print(f"   Initial params shape: {jax.tree_util.tree_map(lambda x: x.shape, initial_params)}")

        # Update state.step to 5 (simulating 5 training steps)
        trainer.state = trainer.state.replace(step=5)

        # Save checkpoint
        print(f"\n2. Saving checkpoint at step 5...")
        trainer.save_checkpoint(5)

        ckpt_dir = Path(tmpdir) / "checkpoints" / "step_5"
        assert ckpt_dir.exists(), f"Checkpoint not found at {ckpt_dir}"
        print(f"   ✓ Checkpoint saved to {ckpt_dir}")

        # Modify state to simulate training
        print(f"\n3. Modifying state to simulate training...")
        # Update step
        trainer.state = trainer.state.replace(step=10)
        # Modify params slightly
        new_params = jax.tree_util.tree_map(lambda x: x + 1.0, initial_params)
        trainer.state = trainer.state.replace(params=new_params)

        modified_step = int(trainer.state.step)
        print(f"   Modified step: {modified_step}")
        assert modified_step != initial_step, "Step should have changed"

        # Create new config with checkpoint path
        print(f"\n4. Loading checkpoint...")
        # Just modify the original config
        config.resume_from_checkpoint = str(ckpt_dir)
        load_config = config

        # Create new trainer and load
        model2 = GPT(load_config)
        trainer2 = Trainer(model=model2, config=load_config, evaluators=None, logger=None, seed=42)

        # Load checkpoint (happens in train() but we'll call it directly)
        loaded_step = trainer2.load_checkpoint()

        print(f"   ✓ Checkpoint loaded from step {loaded_step}")

        # Verify loaded state
        print(f"\n5. Verifying loaded state...")
        assert loaded_step == 5, f"Expected step 5, got {loaded_step}"
        assert int(trainer2.state.step) == 5, f"Expected state.step=5, got {trainer2.state.step}"

        # Check params match original
        loaded_params = trainer2.state.params

        # Compare a few params
        def check_equal(x, y):
            return jnp.allclose(x, y, rtol=1e-5)

        params_match = jax.tree_util.tree_map(check_equal, initial_params, loaded_params)
        all_match = all(jax.tree_util.tree_leaves(params_match))

        if all_match:
            print(f"   ✓ Loaded params match saved params")
        else:
            print(f"   ✗ Params don't match!")
            return False

        # Test that loaded state has correct attributes
        print(f"\n6. Checking TrainState attributes...")
        assert hasattr(trainer2.state, 'params'), "State should have params attribute"
        assert hasattr(trainer2.state, 'step'), "State should have step attribute"
        assert hasattr(trainer2.state, 'opt_state'), "State should have opt_state attribute"
        assert hasattr(trainer2.state, 'tx'), "State should have tx attribute"
        print(f"   ✓ All TrainState attributes present")

        print(f"\n{'=' * 80}")
        print(f"✓ ALL TESTS PASSED")
        print(f"{'=' * 80}")
        return True


if __name__ == "__main__":
    try:
        success = test_checkpoint_save_load()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"✗ TEST FAILED")
        print(f"{'=' * 80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
