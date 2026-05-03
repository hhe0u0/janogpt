"""Test next-token prediction pipeline from data loading to loss computation."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from janogpt import Config, GPT
from janogpt.utils import DummyDataLoader


class TestNextTokenPrediction:
    """Test the complete next-token prediction pipeline."""

    def test_data_loader_to_loss_pipeline(self):
        """Test that data loader format works correctly with loss computation.

        This test verifies the critical contract:
        1. Data loader provides input_ids of shape (batch, seq_len)
        2. Loss computation shifts internally: logits[:, :-1] predicts tokens[:, 1:]
        3. This implements next-token prediction correctly
        """
        # Create tiny model and data
        config = Config(
            num_blocks=1,
            emb_dim=32,
            num_heads=2,
            seq_len=8,
            voc_size=100,
            dropout_prob=0.0,
        )

        model = GPT(config)
        rng = jax.random.key(42)

        # Initialize model
        dummy_input = jnp.zeros((1, config.seq_len), dtype=jnp.uint16)
        params = model.init(
            {"params": rng, "dropout": rng}, dummy_input, inference=True
        )["params"]

        # Get batch from data loader
        loader = DummyDataLoader(batch_size=2, seq_len=8, voc_size=100)
        batch = next(iter(loader))

        # Convert to JAX
        input_ids = jnp.array(batch["input_ids"])

        # Forward pass
        logits = model.apply({"params": params}, input_ids, inference=True)

        # Verify shape
        assert logits.shape == (2, 8, 100)  # (batch, seq_len, vocab)

        # Shift for next-token prediction (what trainer does)
        shift_logits = logits[:, :-1, :]  # (2, 7, 100) - predictions
        shift_labels = input_ids[:, 1:]  # (2, 7) - targets

        # Verify shapes
        assert shift_logits.shape == (2, 7, 100)
        assert shift_labels.shape == (2, 7)

        # Verify targets are indeed next tokens
        for batch_idx in range(2):
            for pos in range(7):
                # Position pos+1 in input should be target for position pos
                expected_target = input_ids[batch_idx, pos + 1]
                actual_target = shift_labels[batch_idx, pos]
                assert expected_target == actual_target

        # Compute loss (should not error)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        ).mean()

        # Loss should be finite and positive
        assert jnp.isfinite(loss)
        assert loss > 0

    def test_next_token_prediction_with_known_sequence(self):
        """Test next-token prediction with a known sequence.

        Create a simple sequence where we can verify the targets manually.
        """
        config = Config(
            num_blocks=1,
            emb_dim=32,
            num_heads=2,
            seq_len=5,
            voc_size=10,
            dropout_prob=0.0,
        )

        model = GPT(config)
        rng = jax.random.key(42)

        # Initialize
        dummy_input = jnp.zeros((1, config.seq_len), dtype=jnp.uint16)
        params = model.init(
            {"params": rng, "dropout": rng}, dummy_input, inference=True
        )["params"]

        # Create known sequence: [0, 1, 2, 3, 4]
        input_ids = jnp.array([[0, 1, 2, 3, 4]], dtype=jnp.int32)

        # Forward pass
        logits = model.apply({"params": params}, input_ids, inference=True)

        # Shift for prediction
        shift_logits = logits[:, :-1, :]  # Predictions for positions 0-3
        shift_labels = input_ids[:, 1:]  # Targets: tokens 1-4

        # Verify manually:
        # Position 0 should predict token 1
        # Position 1 should predict token 2
        # Position 2 should predict token 3
        # Position 3 should predict token 4
        expected_targets = jnp.array([[1, 2, 3, 4]])
        assert jnp.array_equal(shift_labels, expected_targets)

        # Verify predictions are for the right positions
        assert shift_logits.shape == (1, 4, 10)  # 4 predictions

    def test_causal_masking_prevents_future_leakage(self):
        """Test that causal masking prevents model from seeing future tokens.

        This is critical for next-token prediction - model at position i
        should only see tokens 0 through i, not i+1 onwards.
        """
        config = Config(
            num_blocks=2,
            emb_dim=64,
            num_heads=4,
            seq_len=8,
            voc_size=100,
            dropout_prob=0.0,
        )

        model = GPT(config)
        rng = jax.random.key(42)

        # Initialize
        dummy_input = jnp.zeros((1, config.seq_len), dtype=jnp.uint16)
        params = model.init(
            {"params": rng, "dropout": rng}, dummy_input, inference=True
        )["params"]

        # Create two sequences that differ only in the last token
        seq1 = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=jnp.int32)
        seq2 = jnp.array([[0, 1, 2, 3, 4, 5, 6, 99]], dtype=jnp.int32)

        # Forward pass
        logits1 = model.apply({"params": params}, seq1, inference=True)
        logits2 = model.apply({"params": params}, seq2, inference=True)

        # Predictions for positions 0-6 should be IDENTICAL
        # (because they don't see position 7 due to causal masking)
        predictions_1to6_seq1 = logits1[0, :7, :]
        predictions_1to6_seq2 = logits2[0, :7, :]

        assert jnp.allclose(
            predictions_1to6_seq1, predictions_1to6_seq2, rtol=1e-5
        ), "Causal masking failed - earlier positions can see future tokens!"

        # Only prediction at position 6 should differ (predicting position 7)
        # But wait - prediction at position 6 predicts token at position 7,
        # so even that should be identical since both see [0,1,2,3,4,5,6]

        # Let's check position 7 - this predicts beyond sequence (not used in training)
        # Actually, the difference should only appear when we change earlier tokens

    def test_position_independence_of_predictions(self):
        """Test that prediction at position i only depends on tokens 0..i."""
        config = Config(
            num_blocks=2,
            emb_dim=64,
            num_heads=4,
            seq_len=8,
            voc_size=100,
            dropout_prob=0.0,
        )

        model = GPT(config)
        rng = jax.random.key(42)

        dummy_input = jnp.zeros((1, config.seq_len), dtype=jnp.uint16)
        params = model.init(
            {"params": rng, "dropout": rng}, dummy_input, inference=True
        )["params"]

        # Prefix: tokens we care about
        prefix = jnp.array([0, 1, 2, 3], dtype=jnp.int32)

        # Create two sequences with same prefix but different suffixes
        seq1 = jnp.array([[0, 1, 2, 3, 10, 11, 12, 13]], dtype=jnp.int32)
        seq2 = jnp.array([[0, 1, 2, 3, 99, 88, 77, 66]], dtype=jnp.int32)

        logits1 = model.apply({"params": params}, seq1, inference=True)
        logits2 = model.apply({"params": params}, seq2, inference=True)

        # Predictions at positions 0-3 should be identical
        # (they only see prefix [0,1,2,3] due to causal masking)
        for pos in range(4):
            pred1 = logits1[0, pos, :]
            pred2 = logits2[0, pos, :]
            assert jnp.allclose(pred1, pred2, rtol=1e-5), (
                f"Prediction at position {pos} differs! "
                f"Causal masking may be leaking future information."
            )
