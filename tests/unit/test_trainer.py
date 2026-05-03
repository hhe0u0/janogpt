"""Unit tests for Trainer."""

import pytest
import jax
import jax.numpy as jnp
import optax

from janogpt import GPT, Config, Trainer
from janogpt.utils import DummyDataLoader


@pytest.fixture
def tiny_config():
    """Tiny config for fast testing."""
    return Config(
        dropout_prob=0.0,
        num_blocks=2,
        emb_dim=64,
        num_heads=2,
        seq_len=16,
        vocab_size=256,
        max_steps=10,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=1e-3,
        warmup_steps=2,
        eval_interval=5,
        log_interval=2,
    )


@pytest.fixture
def tiny_model(tiny_config):
    """Tiny GPT model."""
    return GPT(tiny_config)


@pytest.fixture
def dummy_dataloader(tiny_config):
    """Dummy data loader."""
    return DummyDataLoader(
        batch_size=tiny_config.micro_batch_size * tiny_config.gradient_accumulation_steps,
        seq_len=tiny_config.seq_len,
        vocab_size=tiny_config.voc_size,
    )


class TestTrainerInit:
    """Test trainer initialization."""

    def test_trainer_init_single_device(self, tiny_model, tiny_config):
        """Test trainer initializes on single device."""
        trainer = Trainer(tiny_model, tiny_config, seed=42)

        assert trainer.model == tiny_model
        assert trainer.config == tiny_config
        assert trainer.num_devices == jax.local_device_count()
        assert trainer.state is not None
        assert trainer.state.step == 0

    def test_trainer_creates_train_state(self, tiny_model, tiny_config):
        """Test trainer creates TrainState with params and optimizer."""
        trainer = Trainer(tiny_model, tiny_config, seed=42)

        # Check TrainState fields
        assert hasattr(trainer.state, 'params')
        assert hasattr(trainer.state, 'tx')
        assert hasattr(trainer.state, 'opt_state')
        assert hasattr(trainer.state, 'step')

        # Check params are initialized
        assert trainer.state.params is not None
        assert isinstance(trainer.state.params, dict)

    def test_trainer_with_evaluators(self, tiny_model, tiny_config, dummy_dataloader):
        """Test trainer accepts evaluators."""
        from janogpt.logger import DatasetEvaluator

        evaluator = DatasetEvaluator(
            dataloader=dummy_dataloader,
            eval_iters=2,
            name="test",
        )

        trainer = Trainer(
            tiny_model,
            tiny_config,
            evaluators=[evaluator],
            seed=42,
        )

        assert len(trainer.evaluators) == 1
        assert trainer.evaluators[0] == evaluator

    def test_trainer_with_logger(self, tiny_model, tiny_config):
        """Test trainer accepts logger."""
        from janogpt.logger import ConsoleLogger

        logger = ConsoleLogger()
        trainer = Trainer(
            tiny_model,
            tiny_config,
            logger=logger,
            seed=42,
        )

        assert trainer.logger == logger


class TestLearningRateSchedule:
    """Test learning rate schedule."""

    def test_warmup_cosine_decay(self, tiny_model, tiny_config):
        """Test learning rate follows warmup + cosine decay."""
        trainer = Trainer(tiny_model, tiny_config, seed=42)

        # Get LR at different steps
        lr_step0 = trainer.get_learning_rate(0)
        lr_warmup = trainer.get_learning_rate(tiny_config.warmup_steps // 2)
        lr_peak = trainer.get_learning_rate(tiny_config.warmup_steps)
        lr_end = trainer.get_learning_rate(tiny_config.max_steps)

        # Check warmup phase
        assert lr_step0 < lr_warmup < lr_peak
        assert abs(lr_peak - tiny_config.learning_rate) < 1e-6

        # Check decay phase
        assert lr_end < lr_peak
        assert lr_end >= tiny_config.min_learning_rate

    def test_no_schedule(self, tiny_model):
        """Test constant learning rate when schedule disabled."""
        config = Config(
            dropout_prob=0.0,
            num_blocks=2,
            emb_dim=64,
            num_heads=2,
            seq_len=16,
            vocab_size=256,
            max_steps=10,
            learning_rate=1e-3,
            use_lr_schedule=False,
        )

        trainer = Trainer(tiny_model, config, seed=42)

        # All steps should have same LR
        lr_step0 = trainer.get_learning_rate(0)
        lr_step5 = trainer.get_learning_rate(5)
        lr_step10 = trainer.get_learning_rate(10)

        assert abs(lr_step0 - config.learning_rate) < 1e-6
        assert abs(lr_step5 - config.learning_rate) < 1e-6
        assert abs(lr_step10 - config.learning_rate) < 1e-6


class TestTrainStep:
    """Test training step."""

    def test_train_step_single_device(self, tiny_model, tiny_config, dummy_dataloader):
        """Test single training step on single device."""
        trainer = Trainer(tiny_model, tiny_config, seed=42)

        # Get batch
        batch = next(iter(dummy_dataloader))
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}

        # Initial loss
        initial_step = trainer.state.step

        # Train step
        metrics = trainer._train_step(batch_jax)

        # Check metrics
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert 'grad_norm' in metrics
        assert 'learning_rate' in metrics

        # Check loss is finite
        assert jnp.isfinite(metrics['loss'])
        assert metrics['loss'] > 0

        # Check step incremented
        assert trainer.state.step == initial_step + 1

    def test_loss_decreases_on_dummy_data(self, tiny_model, tiny_config, dummy_dataloader):
        """Test loss decreases on fixed dummy data (overfitting)."""
        trainer = Trainer(tiny_model, tiny_config, seed=42)

        # Get fixed batch
        batch = next(iter(dummy_dataloader))
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}

        # Train multiple steps on same batch
        losses = []
        for _ in range(5):
            metrics = trainer._train_step(batch_jax)
            losses.append(float(metrics['loss']))

        # Loss should decrease (overfitting)
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


class TestEvaluation:
    """Test evaluation."""

    def test_eval_step(self, tiny_model, tiny_config, dummy_dataloader):
        """Test evaluation step is deterministic."""
        trainer = Trainer(tiny_model, tiny_config, seed=42)

        # Get batch
        batch = next(iter(dummy_dataloader))
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}

        # Eval twice
        loss1 = trainer._eval_step(batch_jax)
        loss2 = trainer._eval_step(batch_jax)

        # Should be identical (deterministic)
        assert jnp.allclose(loss1, loss2)

    def test_evaluate(self, tiny_model, tiny_config, dummy_dataloader):
        """Test full evaluation over multiple batches."""
        trainer = Trainer(tiny_model, tiny_config, seed=42)

        eval_loss = trainer.evaluate(dummy_dataloader, eval_iters=3)

        assert isinstance(eval_loss, float)
        assert jnp.isfinite(eval_loss)
        assert eval_loss > 0


class TestDeviceSupport:
    """Test device detection and support."""

    def test_auto_detect_devices(self, tiny_model, tiny_config):
        """Test trainer auto-detects available devices."""
        trainer = Trainer(tiny_model, tiny_config, seed=42)

        num_devices = jax.local_device_count()
        assert trainer.num_devices == num_devices

    def test_replicate_unreplicate(self, tiny_model, tiny_config):
        """Test replicate and unreplicate utilities."""
        trainer = Trainer(tiny_model, tiny_config, seed=42)

        # Create test pytree
        test_tree = {'a': jnp.array([1, 2, 3]), 'b': jnp.array([4, 5])}

        # Replicate
        replicated = trainer.replicate(test_tree)

        # Unreplicate
        unreplicated = trainer.unreplicate(replicated)

        # Should match original
        assert jnp.allclose(unreplicated['a'], test_tree['a'])
        assert jnp.allclose(unreplicated['b'], test_tree['b'])
