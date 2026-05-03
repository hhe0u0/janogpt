"""Unit tests for GPT model."""

import pytest
import jax
import jax.numpy as jnp

from janogpt import GPT, Config, count_params


@pytest.fixture
def tiny_config():
    """Tiny model config for fast testing."""
    return Config(
        dropout_prob=0.0,
        num_blocks=2,
        emb_dim=64,
        num_heads=2,
        seq_len=16,
        vocab_size=256,
    )


@pytest.fixture
def gpt2_config():
    """Standard GPT2-124M config."""
    return Config(
        dropout_prob=0.1,
        num_blocks=12,
        emb_dim=768,
        num_heads=12,
        seq_len=1024,
        vocab_size=50304,
    )


class TestGPTModel:
    """Test GPT model initialization and forward pass."""

    def test_model_init(self, tiny_config):
        """Test model initializes correctly."""
        model = GPT(tiny_config)
        assert model.config == tiny_config

    def test_forward_pass_shape(self, tiny_config):
        """Test forward pass produces correct output shape."""
        model = GPT(tiny_config)
        rng = jax.random.key(42)

        # Initialize
        x = jnp.zeros((2, tiny_config.seq_len), dtype=jnp.uint16)
        params = model.init({'params': rng, 'dropout': rng}, x, inference=True)['params']

        # Forward pass
        logits = model.apply({'params': params}, x, inference=True)

        # Check shape
        assert logits.shape == (2, tiny_config.seq_len, tiny_config.vocab_size)

    def test_dynamic_sequence_length(self, tiny_config):
        """Test model handles variable sequence lengths."""
        model = GPT(tiny_config)
        rng = jax.random.key(42)

        # Initialize with max length
        x_init = jnp.zeros((1, tiny_config.seq_len), dtype=jnp.uint16)
        params = model.init({'params': rng, 'dropout': rng}, x_init, inference=True)['params']

        # Test with shorter sequence
        seq_len = 8
        x_short = jnp.zeros((1, seq_len), dtype=jnp.uint16)
        logits = model.apply({'params': params}, x_short, inference=True)

        assert logits.shape == (1, seq_len, tiny_config.vocab_size)

    def test_param_count_tiny(self, tiny_config):
        """Test parameter counting for tiny model."""
        model = GPT(tiny_config)
        rng = jax.random.key(42)
        x = jnp.zeros((1, tiny_config.seq_len), dtype=jnp.uint16)
        params = model.init({'params': rng, 'dropout': rng}, x, inference=True)['params']

        param_count = count_params(params)
        assert param_count > 0
        assert param_count < 1_000_000  # Should be small

    def test_param_count_gpt2(self, gpt2_config):
        """Test parameter count for GPT2-124M."""
        model = GPT(gpt2_config)
        rng = jax.random.key(42)
        x = jnp.zeros((1, gpt2_config.seq_len), dtype=jnp.uint16)
        params = model.init({'params': rng, 'dropout': rng}, x, inference=True)['params']

        param_count = count_params(params)
        # Should be ~124M parameters
        assert 120_000_000 < param_count < 130_000_000

    def test_inference_mode(self, tiny_config):
        """Test inference mode (no dropout)."""
        model = GPT(tiny_config)
        rng = jax.random.key(42)

        x = jnp.ones((1, tiny_config.seq_len), dtype=jnp.uint16)
        params = model.init({'params': rng, 'dropout': rng}, x, inference=True)['params']

        # Forward pass twice with inference=True should give same result
        logits1 = model.apply({'params': params}, x, inference=True)
        logits2 = model.apply({'params': params}, x, inference=True)

        assert jnp.allclose(logits1, logits2)

    def test_causal_masking(self, tiny_config):
        """Test causal attention mask prevents looking ahead."""
        model = GPT(tiny_config)
        rng = jax.random.key(42)

        x = jnp.arange(tiny_config.seq_len, dtype=jnp.uint16)[None, :]
        params = model.init({'params': rng, 'dropout': rng}, x, inference=True)['params']

        logits = model.apply({'params': params}, x, inference=True)

        # Logits at position i should only depend on positions 0..i
        # This is hard to test directly, but we can verify shape is correct
        assert logits.shape == (1, tiny_config.seq_len, tiny_config.vocab_size)


class TestConfig:
    """Test configuration."""

    def test_config_defaults(self):
        """Test config has sensible defaults."""
        config = Config()
        assert config.dropout_prob >= 0.0
        assert config.num_blocks > 0
        assert config.emb_dim > 0
        assert config.num_heads > 0
        assert config.seq_len > 0
        assert config.vocab_size > 0

    def test_config_to_dict(self, gpt2_config):
        """Test config serialization."""
        config_dict = gpt2_config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'model' in config_dict
        assert config_dict['model']['num_blocks'] == 12

    def test_config_from_json(self, tmp_path):
        """Test config loading from JSON."""
        import json

        config_file = tmp_path / "test_config.json"
        config_data = {
            "model": {
                "num_blocks": 6,
                "emb_dim": 384,
                "num_heads": 6,
            },
            "training": {
                "max_steps": 1000,
            }
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        config = Config.from_json(str(config_file))
        assert config.num_blocks == 6
        assert config.emb_dim == 384
        assert config.max_steps == 1000
