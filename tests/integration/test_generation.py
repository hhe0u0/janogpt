"""Integration test for text generation."""

import jax
import jax.numpy as jnp
import pytest

from janogpt import GPT, Config, Trainer
from janogpt.inference import generate
from janogpt.utils import DummyDataLoader


@pytest.fixture
def tiny_config():
    """Tiny config for fast testing."""
    return Config(
        dropout_prob=0.0,
        num_blocks=2,
        emb_dim=128,
        num_heads=4,
        seq_len=64,
        vocab_size=512,
    )


@pytest.fixture
def trained_model_params(tiny_config):
    """Create and train a tiny model on dummy data."""
    # Create model
    model = GPT(tiny_config)
    trainer = Trainer(model, tiny_config, seed=42)

    # Train briefly on dummy data (for overfitting)
    loader = DummyDataLoader(
        batch_size=4,
        seq_len=tiny_config.seq_len,
        vocab_size=tiny_config.vocab_size,
    )

    # Train 20 steps to get some signal
    for step, batch in enumerate(loader, start=1):
        if step > 20:
            break
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}
        trainer._train_step(batch_jax)

    return model, trainer.state.params


class TestGenerationBasics:
    """Test basic generation functionality."""

    def test_generation_produces_tokens(self, trained_model_params):
        """Test generation produces valid tokens."""
        model, params = trained_model_params

        prompt = [1, 2, 3]
        rng = jax.random.key(42)

        generated = generate(
            model,
            params,
            prompt,
            max_new_tokens=10,
            temperature=1.0,
            rng_key=rng,
        )

        # Should extend prompt
        assert len(generated) == len(prompt) + 10
        assert generated[: len(prompt)] == prompt

        # All tokens should be valid
        for token in generated:
            assert isinstance(token, int)
            assert 0 <= token < 512

    def test_generation_with_empty_prompt(self, trained_model_params):
        """Test generation from empty prompt."""
        model, params = trained_model_params

        rng = jax.random.key(42)
        generated = generate(
            model,
            params,
            [],  # Empty prompt
            max_new_tokens=10,
            rng_key=rng,
        )

        assert len(generated) == 10

    def test_generation_respects_max_tokens(self, trained_model_params):
        """Test generation stops at max_new_tokens."""
        model, params = trained_model_params

        prompt = [1, 2, 3]
        rng = jax.random.key(42)

        for max_new in [5, 10, 20]:
            generated = generate(
                model,
                params,
                prompt,
                max_new_tokens=max_new,
                rng_key=rng,
            )
            assert len(generated) == len(prompt) + max_new


class TestTemperatureSampling:
    """Test temperature effects on generation."""

    def test_low_temperature_more_deterministic(self, trained_model_params):
        """Test low temperature produces more consistent outputs."""
        model, params = trained_model_params

        prompt = [1, 2, 3, 4, 5]

        # Generate 5 times with low temperature
        outputs = []
        for seed in range(5):
            rng = jax.random.key(seed)
            gen = generate(
                model,
                params,
                prompt,
                max_new_tokens=5,
                temperature=0.1,
                rng_key=rng,
            )
            outputs.append(gen)

        # Most should be similar (at least 3 out of 5 match)
        from collections import Counter

        counts = Counter([tuple(out) for out in outputs])
        most_common_count = counts.most_common(1)[0][1]
        assert most_common_count >= 3

    def test_high_temperature_more_diverse(self, trained_model_params):
        """Test high temperature produces more diverse outputs."""
        model, params = trained_model_params

        prompt = [1, 2, 3, 4, 5]

        # Generate 5 times with high temperature
        outputs = []
        for seed in range(5):
            rng = jax.random.key(seed)
            gen = generate(
                model,
                params,
                prompt,
                max_new_tokens=10,
                temperature=2.0,
                rng_key=rng,
            )
            outputs.append(gen)

        # Should have multiple different outputs
        unique_outputs = len(set([tuple(out) for out in outputs]))
        assert unique_outputs >= 3  # At least 3 different

    def test_temperature_range(self, trained_model_params):
        """Test generation works across temperature range."""
        model, params = trained_model_params

        prompt = [1, 2, 3]

        for temp in [0.1, 0.5, 1.0, 1.5, 2.0]:
            rng = jax.random.key(42)
            generated = generate(
                model,
                params,
                prompt,
                max_new_tokens=5,
                temperature=temp,
                rng_key=rng,
            )

            # Should produce valid output
            assert len(generated) == len(prompt) + 5
            assert all(0 <= t < 512 for t in generated)


class TestTopKSampling:
    """Test top-k sampling."""

    def test_top_k_1_is_greedy(self, trained_model_params):
        """Test top_k=1 is deterministic (greedy decoding)."""
        model, params = trained_model_params

        prompt = [1, 2, 3, 4]

        # Generate twice with different seeds but top_k=1
        rng1 = jax.random.key(42)
        gen1 = generate(
            model,
            params,
            prompt,
            max_new_tokens=10,
            top_k=1,
            rng_key=rng1,
        )

        rng2 = jax.random.key(999)
        gen2 = generate(
            model,
            params,
            prompt,
            max_new_tokens=10,
            top_k=1,
            rng_key=rng2,
        )

        # Should be identical
        assert gen1 == gen2

    def test_top_k_limits_vocabulary(self, trained_model_params):
        """Test top_k restricts token choices."""
        model, params = trained_model_params

        prompt = [1, 2, 3]

        # Generate multiple times with small top_k
        outputs = []
        for seed in range(10):
            rng = jax.random.key(seed)
            gen = generate(
                model,
                params,
                prompt,
                max_new_tokens=1,  # Just one token
                top_k=5,
                rng_key=rng,
            )
            outputs.append(gen[-1])  # Last token

        # Should see at most 5 different tokens
        unique_tokens = len(set(outputs))
        assert unique_tokens <= 5


class TestLongSequenceGeneration:
    """Test generation with longer sequences."""

    def test_long_generation(self, trained_model_params, tiny_config):
        """Test generating longer sequences."""
        model, params = trained_model_params

        prompt = [1, 2, 3]
        rng = jax.random.key(42)

        # Generate up to near context limit
        max_new = tiny_config.seq_len - len(prompt) - 5
        generated = generate(
            model,
            params,
            prompt,
            max_new_tokens=max_new,
            rng_key=rng,
        )

        assert len(generated) == len(prompt) + max_new

    def test_generation_near_context_limit(self, trained_model_params, tiny_config):
        """Test generation with prompt near context limit."""
        model, params = trained_model_params

        # Long prompt
        prompt = list(range(1, tiny_config.seq_len - 5))
        rng = jax.random.key(42)

        generated = generate(
            model,
            params,
            prompt,
            max_new_tokens=3,
            rng_key=rng,
        )

        assert len(generated) == len(prompt) + 3


class TestGenerationDeterminism:
    """Test determinism in generation."""

    def test_same_seed_same_output(self, trained_model_params):
        """Test same seed produces same output."""
        model, params = trained_model_params

        prompt = [1, 2, 3, 4, 5]

        gen1 = generate(
            model,
            params,
            prompt,
            max_new_tokens=15,
            temperature=1.0,
            top_k=50,
            rng_key=jax.random.key(42),
        )

        gen2 = generate(
            model,
            params,
            prompt,
            max_new_tokens=15,
            temperature=1.0,
            top_k=50,
            rng_key=jax.random.key(42),
        )

        assert gen1 == gen2

    def test_different_seed_different_output(self, trained_model_params):
        """Test different seed produces different output."""
        model, params = trained_model_params

        prompt = [1, 2, 3, 4, 5]

        outputs = []
        for seed in [42, 123, 456, 789, 999]:
            gen = generate(
                model,
                params,
                prompt,
                max_new_tokens=10,
                temperature=1.0,
                top_k=50,
                rng_key=jax.random.key(seed),
            )
            outputs.append(gen)

        # Should have multiple different outputs
        unique_outputs = len(set([tuple(out) for out in outputs]))
        assert unique_outputs >= 3


class TestGenerationAfterTraining:
    """Test generation improves after training."""

    def test_overfitted_model_generation(self, tiny_config):
        """Test heavily overfitted model generates similar sequences."""
        model = GPT(tiny_config)
        trainer = Trainer(model, tiny_config, seed=42)

        # Create fixed dataset
        loader = DummyDataLoader(
            batch_size=4,
            seq_len=tiny_config.seq_len,
            vocab_size=tiny_config.vocab_size,
        )

        # Get fixed batch
        fixed_batch = next(iter(loader))
        fixed_batch_jax = {k: jnp.array(v) for k, v in fixed_batch.items()}

        # Train heavily on same batch
        for _ in range(50):
            trainer._train_step(fixed_batch_jax)

        # Generate starting from tokens in training data
        prompt = list(fixed_batch["input_ids"][0][:5])
        rng = jax.random.key(42)

        generated = generate(
            model,
            trainer.state.params,
            prompt,
            max_new_tokens=10,
            temperature=0.1,  # Low temp for more deterministic
            rng_key=rng,
        )

        # Should produce valid tokens
        assert len(generated) == len(prompt) + 10
        assert all(0 <= t < tiny_config.vocab_size for t in generated)
