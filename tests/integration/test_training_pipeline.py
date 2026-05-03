"""Integration test for end-to-end training pipeline."""


import pytest

from janogpt import GPT, Config, Trainer
from janogpt.logger import ConsoleLogger, DatasetEvaluator
from janogpt.utils import DummyDataLoader


@pytest.fixture
def smoke_config():
    """Smoke test config for fast integration testing."""
    return Config(
        # Tiny model
        dropout_prob=0.0,
        num_blocks=2,
        emb_dim=128,
        num_heads=2,
        seq_len=64,
        voc_size=512,
        # Short training
        max_steps=5,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        warmup_steps=2,
        # Frequent logging
        eval_interval=3,
        eval_iters=2,
        log_interval=1,
        # No WandB
        wandb_log=False,
    )


@pytest.fixture
def smoke_dataloaders(smoke_config):
    """Create train and eval dataloaders."""
    train_loader = DummyDataLoader(
        batch_size=smoke_config.micro_batch_size * smoke_config.gradient_accumulation_steps,
        seq_len=smoke_config.seq_len,
        voc_size=smoke_config.voc_size,
    )

    eval_loader = DummyDataLoader(
        batch_size=smoke_config.micro_batch_size * smoke_config.gradient_accumulation_steps,
        seq_len=smoke_config.seq_len,
        voc_size=smoke_config.voc_size,
    )

    return train_loader, eval_loader


class TestTrainingPipeline:
    """Test end-to-end training pipeline."""

    def test_smoke_training_run(self, smoke_config, smoke_dataloaders):
        """Test full training pipeline completes without errors."""
        train_loader, eval_loader = smoke_dataloaders

        # Create model
        model = GPT(smoke_config)

        # Create trainer with console logging
        logger = ConsoleLogger()
        evaluator = DatasetEvaluator(
            dataloader=eval_loader,
            eval_iters=smoke_config.eval_iters,
            name="val",
        )

        trainer = Trainer(
            model,
            smoke_config,
            evaluators=[evaluator],
            logger=logger,
            seed=42,
        )

        # Run training - should complete without errors
        trainer.train(train_loader, eval_loader)

        # Verify training ran
        assert trainer.state.step == smoke_config.max_steps

    def test_loss_decreases_on_dummy_data(self, smoke_config, smoke_dataloaders):
        """Test loss decreases when overfitting dummy data."""
        train_loader, _ = smoke_dataloaders

        # Longer training for convergence
        config = smoke_config
        config.max_steps = 20
        config.log_interval = 5

        model = GPT(config)
        trainer = Trainer(model, config, seed=42)

        # Track losses
        initial_loss = None
        final_loss = None

        # Run training
        for step, batch in enumerate(train_loader, start=1):
            if step > config.max_steps:
                break

            import jax.numpy as jnp

            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            metrics = trainer._train_step(batch_jax)

            if step == 1:
                initial_loss = float(metrics["loss"])
            final_loss = float(metrics["loss"])

        # Loss should decrease significantly
        assert final_loss < initial_loss
        assert final_loss < initial_loss * 0.8  # At least 20% reduction

    def test_evaluation_runs_correctly(self, smoke_config, smoke_dataloaders):
        """Test evaluation runs and returns valid loss."""
        _, eval_loader = smoke_dataloaders

        model = GPT(smoke_config)
        trainer = Trainer(model, smoke_config, seed=42)

        eval_loss = trainer.evaluate(eval_loader, eval_iters=3)

        assert isinstance(eval_loss, float)
        assert eval_loss > 0
        import jax.numpy as jnp

        assert jnp.isfinite(eval_loss)

    def test_checkpointing_workflow(self, smoke_config, smoke_dataloaders, tmp_path):
        """Test saving and loading checkpoints."""
        train_loader, _ = smoke_dataloaders

        # Set output dir to temp path
        config = smoke_config
        config.output_dir = str(tmp_path / "output")
        config.save_interval = 3
        config.max_steps = 5

        # Train and save checkpoint
        model = GPT(config)
        trainer1 = Trainer(model, config, seed=42)

        # Train 3 steps and save
        for step, batch in enumerate(train_loader, start=1):
            if step > 3:
                break
            import jax.numpy as jnp

            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            trainer1._train_step(batch_jax)

        trainer1.save_checkpoint(3)

        # Load checkpoint in new trainer
        config2 = smoke_config
        config2.output_dir = str(tmp_path / "output")
        config2.resume_from_checkpoint = str(tmp_path / "output" / "checkpoints" / "step_3")

        model2 = GPT(config2)
        trainer2 = Trainer(model2, config2, seed=42)
        loaded_step = trainer2.load_checkpoint(3)

        assert loaded_step == 3

        # Parameters should match
        import jax.numpy as jnp

        for k in trainer1.state.params.keys():
            if trainer1.num_devices > 1:
                # Unreplicate if multi-device
                p1 = trainer1.unreplicate(trainer1.state.params)[k]
                p2 = trainer2.unreplicate(trainer2.state.params)[k]
            else:
                p1 = trainer1.state.params[k]
                p2 = trainer2.state.params[k]

            # Compare recursively for nested dicts
            def compare_nested(x, y):
                if isinstance(x, dict):
                    for key in x:
                        compare_nested(x[key], y[key])
                else:
                    assert jnp.allclose(x, y, rtol=1e-5)

            compare_nested(p1, p2)


class TestMultiDeviceTraining:
    """Test multi-device training (if available)."""

    def test_auto_detect_devices(self, smoke_config):
        """Test trainer auto-detects available devices."""
        import jax

        model = GPT(smoke_config)
        trainer = Trainer(model, smoke_config, seed=42)

        assert trainer.num_devices == jax.local_device_count()

    @pytest.mark.skipif(
        lambda: __import__("jax").local_device_count() < 2, reason="Requires multiple devices"
    )
    def test_multi_device_training(self, smoke_config, smoke_dataloaders):
        """Test training works on multiple devices."""
        train_loader, _ = smoke_dataloaders

        model = GPT(smoke_config)
        trainer = Trainer(model, smoke_config, seed=42)

        # Should work with pmap on multiple devices
        for step, batch in enumerate(train_loader, start=1):
            if step > 3:
                break
            import jax.numpy as jnp

            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            metrics = trainer._train_step(batch_jax)

            assert "loss" in metrics
            assert jnp.isfinite(metrics["loss"])


class TestConfigIntegration:
    """Test training with JSON config files."""

    def test_load_config_and_train(self, tmp_path):
        """Test loading config from JSON and training."""
        import json

        # Create config file
        config_file = tmp_path / "test_config.json"
        config_data = {
            "model": {
                "num_blocks": 2,
                "emb_dim": 64,
                "num_heads": 2,
                "seq_len": 32,
                "voc_size": 256,
            },
            "training": {
                "max_steps": 3,
                "micro_batch_size": 2,
            },
            "data": {
                "dataloader_class": "DummyDataLoader",
            },
            "wandb": {
                "enabled": False,
            },
        }

        with open(config_file, "w") as f:
            json.dump(config_data, f)

        # Load config
        config = Config.from_json(str(config_file))

        # Create data loader
        from janogpt.utils import DummyDataLoader

        train_loader = DummyDataLoader(
            batch_size=config.micro_batch_size,
            seq_len=config.seq_len,
            voc_size=config.voc_size,
        )

        # Train
        model = GPT(config)
        trainer = Trainer(model, config, seed=42)

        for step, batch in enumerate(train_loader, start=1):
            if step > config.max_steps:
                break
            import jax.numpy as jnp

            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            trainer._train_step(batch_jax)

        assert trainer.state.step == config.max_steps
