"""Unit tests for Config."""

import pytest
import json
from pathlib import Path

from janogpt.config import Config


class TestConfigDefaults:
    """Test config default values."""

    def test_default_config(self):
        """Test config has sensible defaults."""
        config = Config()

        # Model architecture
        assert config.dropout_prob >= 0.0
        assert config.num_blocks > 0
        assert config.emb_dim > 0
        assert config.num_heads > 0
        assert config.seq_len > 0
        assert config.voc_size > 0

        # Training hyperparameters
        assert config.learning_rate > 0
        assert config.max_steps > 0
        assert config.micro_batch_size > 0


class TestConfigSerialization:
    """Test config serialization and deserialization."""

    def test_to_dict(self):
        """Test config to dict conversion."""
        config = Config(
            num_blocks=6,
            emb_dim=384,
            num_heads=6,
            learning_rate=3e-4,
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict['num_blocks'] == 6
        assert config_dict['emb_dim'] == 384
        assert config_dict['num_heads'] == 6
        assert config_dict['learning_rate'] == 3e-4

    def test_from_json(self, tmp_path):
        """Test loading config from JSON file."""
        config_file = tmp_path / "test_config.json"
        config_data = {
            "model": {
                "num_blocks": 6,
                "emb_dim": 384,
                "num_heads": 6,
            },
            "optimizer": {
                "learning_rate": 3e-4,
            },
            "training": {
                "max_steps": 10000,
                "micro_batch_size": 8,
            },
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        config = Config.from_json(str(config_file))

        assert config.num_blocks == 6
        assert config.emb_dim == 384
        assert config.num_heads == 6
        assert config.learning_rate == 3e-4
        assert config.max_steps == 10000
        assert config.micro_batch_size == 8

    def test_from_json_with_data_section(self, tmp_path):
        """Test loading config with data section."""
        config_file = tmp_path / "test_config.json"
        config_data = {
            "model": {
                "num_blocks": 2,
            },
            "data": {
                "data_dir": "data/custom",
                "dataloader_class": "FileDataLoader",
            },
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        config = Config.from_json(str(config_file))

        assert config.num_blocks == 2
        assert config.data_dir == "data/custom"
        assert config.dataloader_class == "FileDataLoader"

    def test_save_json(self, tmp_path):
        """Test saving config to JSON file."""
        config = Config(
            num_blocks=6,
            emb_dim=384,
            max_steps=5000,
        )

        config_file = tmp_path / "saved_config.json"
        config.save_json(str(config_file))

        # Load back and verify
        assert config_file.exists()
        with open(config_file) as f:
            loaded = json.load(f)

        assert loaded['num_blocks'] == 6
        assert loaded['emb_dim'] == 384
        assert loaded['max_steps'] == 5000

    def test_round_trip(self, tmp_path):
        """Test save and load produces same config."""
        config = Config(
            num_blocks=8,
            emb_dim=512,
            num_heads=8,
            learning_rate=6e-4,
            max_steps=100000,
        )

        # Save
        config_file = tmp_path / "round_trip.json"
        config.save_json(str(config_file))

        # Load
        loaded_config = Config.from_json(str(config_file))

        # Verify
        assert loaded_config.num_blocks == config.num_blocks
        assert loaded_config.emb_dim == config.emb_dim
        assert loaded_config.num_heads == config.num_heads
        assert loaded_config.learning_rate == config.learning_rate
        assert loaded_config.max_steps == config.max_steps


class TestConfigValidation:
    """Test config validation."""

    def test_ff_dim_default(self):
        """Test ff_dim defaults to 4 * emb_dim."""
        config = Config(emb_dim=768)
        assert config.ff_dim == 4 * 768

    def test_ff_dim_override(self):
        """Test ff_dim can be overridden."""
        config = Config(emb_dim=768, ff_dim=2048)
        assert config.ff_dim == 2048

    def test_min_learning_rate_default(self):
        """Test min_learning_rate defaults to 10% of learning_rate."""
        config = Config(learning_rate=6e-4)
        assert config.min_learning_rate == 6e-5

    def test_gradient_accumulation_default(self):
        """Test gradient accumulation defaults to 1."""
        config = Config()
        assert config.gradient_accumulation_steps == 1


class TestConfigComputedProperties:
    """Test computed properties."""

    def test_effective_batch_size(self):
        """Test effective batch size calculation."""
        config = Config(
            micro_batch_size=4,
            gradient_accumulation_steps=8,
        )

        # For single device (test environment)
        num_devices = 1
        effective_batch = config.micro_batch_size * config.gradient_accumulation_steps * num_devices
        expected = 4 * 8 * 1

        assert effective_batch == expected

    def test_tokens_per_step(self):
        """Test tokens per step calculation."""
        config = Config(
            micro_batch_size=4,
            gradient_accumulation_steps=16,
            seq_len=1024,
        )

        # For 8 devices: 4 * 16 * 8 * 1024 = 524,288 ≈ 0.5M
        num_devices = 8
        tokens_per_step = (
            config.micro_batch_size
            * config.gradient_accumulation_steps
            * num_devices
            * config.seq_len
        )

        assert tokens_per_step == 524_288


class TestConfigEdgeCases:
    """Test edge cases."""

    def test_from_json_missing_sections(self, tmp_path):
        """Test loading config with missing sections uses defaults."""
        config_file = tmp_path / "minimal_config.json"
        config_data = {
            "model": {
                "num_blocks": 3,
            },
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        config = Config.from_json(str(config_file))

        # Overridden value
        assert config.num_blocks == 3

        # Default values should still be present
        assert config.emb_dim > 0
        assert config.learning_rate > 0

    def test_from_json_nested_overrides(self, tmp_path):
        """Test nested section overrides work correctly."""
        config_file = tmp_path / "nested_config.json"
        config_data = {
            "model": {
                "num_blocks": 12,
                "emb_dim": 768,
            },
            "optimizer": {
                "learning_rate": 6e-4,
                "beta1": 0.9,
                "beta2": 0.95,
            },
            "training": {
                "max_steps": 600000,
            },
            "wandb": {
                "enabled": True,
                "project": "test-project",
            },
        }

        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        config = Config.from_json(str(config_file))

        # Model section
        assert config.num_blocks == 12
        assert config.emb_dim == 768

        # Optimizer section
        assert config.learning_rate == 6e-4
        assert config.beta1 == 0.9
        assert config.beta2 == 0.95

        # Training section
        assert config.max_steps == 600000

        # WandB section
        assert config.wandb_log is True
        assert config.wandb_project == "test-project"
