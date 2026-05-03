"""Unit tests for inference and generation."""

import pytest
import jax
import jax.numpy as jnp

from janogpt import GPT, Config
from janogpt.inference import generate


@pytest.fixture
def tiny_config():
    """Tiny config for fast testing."""
    return Config(
        dropout_prob=0.0,
        num_blocks=2,
        emb_dim=64,
        num_heads=2,
        seq_len=32,
        vocab_size=256,
    )


@pytest.fixture
def tiny_model_and_params(tiny_config):
    """Initialized tiny model with random params."""
    model = GPT(tiny_config)
    rng = jax.random.key(42)
    x = jnp.zeros((1, tiny_config.seq_len), dtype=jnp.uint16)
    params = model.init({'params': rng, 'dropout': rng}, x, inference=True)['params']
    return model, params


class TestGeneration:
    """Test text generation."""

    def test_generate_basic(self, tiny_model_and_params, tiny_config):
        """Test basic generation produces tokens."""
        model, params = tiny_model_and_params

        prompt_tokens = [1, 2, 3]
        rng = jax.random.key(42)

        generated = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=5,
            temperature=1.0,
            rng_key=rng,
        )

        # Should return prompt + new tokens
        assert len(generated) == len(prompt_tokens) + 5
        assert generated[:len(prompt_tokens)] == prompt_tokens

    def test_generate_respects_max_new_tokens(self, tiny_model_and_params):
        """Test generation stops at max_new_tokens."""
        model, params = tiny_model_and_params

        prompt_tokens = [1, 2, 3]
        max_new = 10
        rng = jax.random.key(42)

        generated = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=max_new,
            rng_key=rng,
        )

        assert len(generated) == len(prompt_tokens) + max_new

    def test_generate_temperature_effect(self, tiny_model_and_params):
        """Test temperature affects randomness."""
        model, params = tiny_model_and_params

        prompt_tokens = [1, 2, 3]
        rng = jax.random.key(42)

        # Low temperature (more deterministic)
        rng1, rng2 = jax.random.split(rng)
        gen_low_temp = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=10,
            temperature=0.1,
            rng_key=rng1,
        )

        # High temperature (more random)
        gen_high_temp = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=10,
            temperature=2.0,
            rng_key=rng2,
        )

        # Both should produce valid tokens
        assert all(0 <= t < 256 for t in gen_low_temp)
        assert all(0 <= t < 256 for t in gen_high_temp)

    def test_generate_top_k_filtering(self, tiny_model_and_params):
        """Test top-k filtering limits token choices."""
        model, params = tiny_model_and_params

        prompt_tokens = [1, 2, 3]
        rng = jax.random.key(42)

        # With top_k=1, should be deterministic (greedy)
        rng1, rng2 = jax.random.split(rng)
        gen1 = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=5,
            top_k=1,
            rng_key=rng1,
        )
        gen2 = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=5,
            top_k=1,
            rng_key=rng2,
        )

        # Should produce same output (greedy decoding)
        assert gen1 == gen2

    def test_generate_deterministic_with_same_seed(self, tiny_model_and_params):
        """Test generation is deterministic with same RNG."""
        model, params = tiny_model_and_params

        prompt_tokens = [1, 2, 3]

        gen1 = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=10,
            temperature=1.0,
            rng_key=jax.random.key(42),
        )
        gen2 = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=10,
            temperature=1.0,
            rng_key=jax.random.key(42),
        )

        assert gen1 == gen2

    def test_generate_different_with_different_seed(self, tiny_model_and_params):
        """Test generation differs with different RNG."""
        model, params = tiny_model_and_params

        prompt_tokens = [1, 2, 3]

        gen1 = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=10,
            temperature=1.0,
            rng_key=jax.random.key(42),
        )
        gen2 = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=10,
            temperature=1.0,
            rng_key=jax.random.key(123),
        )

        # Should differ (with high probability)
        assert gen1 != gen2

    def test_generate_handles_empty_prompt(self, tiny_model_and_params):
        """Test generation works with empty prompt."""
        model, params = tiny_model_and_params

        prompt_tokens = []
        rng = jax.random.key(42)

        generated = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=5,
            rng_key=rng,
        )

        assert len(generated) == 5

    def test_generate_handles_long_prompt(self, tiny_model_and_params, tiny_config):
        """Test generation handles prompt near context limit."""
        model, params = tiny_model_and_params

        # Prompt near seq_len
        prompt_tokens = list(range(1, tiny_config.seq_len - 5))
        rng = jax.random.key(42)

        generated = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=3,
            rng_key=rng,
        )

        assert len(generated) == len(prompt_tokens) + 3


class TestGenerationEdgeCases:
    """Test edge cases in generation."""

    def test_generate_respects_vocab_size(self, tiny_model_and_params, tiny_config):
        """Test all generated tokens are within vocabulary."""
        model, params = tiny_model_and_params

        prompt_tokens = [1, 2, 3]
        rng = jax.random.key(42)

        generated = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=20,
            rng_key=rng,
        )

        # All tokens should be valid
        for token in generated:
            assert 0 <= token < tiny_config.voc_size

    def test_generate_with_temperature_zero(self, tiny_model_and_params):
        """Test generation with very low temperature (greedy)."""
        model, params = tiny_model_and_params

        prompt_tokens = [1, 2, 3]
        rng = jax.random.key(42)

        # Very low temperature should be greedy
        generated = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=5,
            temperature=0.01,
            rng_key=rng,
        )

        assert len(generated) == len(prompt_tokens) + 5

    def test_generate_with_high_temperature(self, tiny_model_and_params):
        """Test generation with high temperature."""
        model, params = tiny_model_and_params

        prompt_tokens = [1, 2, 3]
        rng = jax.random.key(42)

        # High temperature should still work
        generated = generate(
            model,
            params,
            prompt_tokens,
            max_new_tokens=5,
            temperature=5.0,
            rng_key=rng,
        )

        assert len(generated) == len(prompt_tokens) + 5
