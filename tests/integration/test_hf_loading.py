"""Integration test for HuggingFace weight loading."""

import jax
import jax.numpy as jnp
import pytest

from janogpt import GPT, Config
from janogpt.inference import generate

# Skip all tests if transformers not available
transformers = pytest.importorskip("transformers")


@pytest.fixture
def gpt2_config():
    """GPT-2 config matching HuggingFace."""
    return Config(
        dropout_prob=0.1,
        num_blocks=12,
        emb_dim=768,
        num_heads=12,
        seq_len=1024,
        voc_size=50257,  # CRITICAL: HF uses 50257, not 50304
    )


@pytest.fixture
def hf_model():
    """Load HuggingFace GPT-2 model."""
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return model


@pytest.fixture
def hf_tokenizer():
    """Load HuggingFace tokenizer."""
    from transformers import GPT2Tokenizer

    return GPT2Tokenizer.from_pretrained("gpt2")


class TestHFWeightLoading:
    """Test loading pretrained weights from HuggingFace."""

    def test_convert_hf_weights(self, gpt2_config):
        """Test converting HF weights to JAX format."""
        from pretrained.huggingface.loader import convert_hf_to_jax

        # Should not raise errors
        jax_params = convert_hf_to_jax("gpt2", gpt2_config)

        # Check structure
        assert isinstance(jax_params, dict)
        assert "Emb_0" in jax_params  # Embeddings
        assert "TransformerBlock_0" in jax_params  # First block

    def test_loaded_params_shape_matches(self, gpt2_config):
        """Test loaded params have correct shapes."""
        from pretrained.huggingface.loader import convert_hf_to_jax

        jax_params = convert_hf_to_jax("gpt2", gpt2_config)

        # Check embedding shapes
        wte = jax_params["Emb_0"]["Embed_0"]["embedding"]
        wpe = jax_params["Emb_0"]["Embed_1"]["embedding"]

        assert wte.shape == (50257, 768)  # Token embeddings
        assert wpe.shape == (1024, 768)  # Position embeddings

    def test_model_forward_with_hf_weights(self, gpt2_config):
        """Test forward pass with loaded HF weights."""
        from pretrained.huggingface.loader import convert_hf_to_jax

        # Load weights
        jax_params = convert_hf_to_jax("gpt2", gpt2_config)

        # Create model
        model = GPT(gpt2_config)

        # Forward pass
        x = jnp.array([[1, 2, 3, 4]], dtype=jnp.uint16)
        logits = model.apply({"params": jax_params}, x, inference=True)

        # Check output shape
        assert logits.shape == (1, 4, 50257)

        # Check logits are finite
        assert jnp.all(jnp.isfinite(logits))


class TestHFGenerationComparison:
    """Compare generation between HF and JAX implementations."""

    def test_generation_produces_text(self, gpt2_config, hf_tokenizer):
        """Test generation produces valid text."""
        from pretrained.huggingface.loader import convert_hf_to_jax

        # Load weights
        jax_params = convert_hf_to_jax("gpt2", gpt2_config)
        model = GPT(gpt2_config)

        # Tokenize prompt
        prompt = "Hello, I am"
        tokens = hf_tokenizer.encode(prompt)

        # Generate
        rng = jax.random.key(42)
        generated_tokens = generate(
            model,
            jax_params,
            tokens,
            max_new_tokens=10,
            temperature=0.8,
            top_k=50,
            rng_key=rng,
        )

        # Decode
        generated_text = hf_tokenizer.decode(generated_tokens)

        # Should be valid text
        assert len(generated_text) > len(prompt)
        assert generated_text.startswith(prompt)

    def test_greedy_generation_deterministic(self, gpt2_config, hf_tokenizer):
        """Test greedy generation is deterministic."""
        from pretrained.huggingface.loader import convert_hf_to_jax

        jax_params = convert_hf_to_jax("gpt2", gpt2_config)
        model = GPT(gpt2_config)

        prompt = "The capital of France is"
        tokens = hf_tokenizer.encode(prompt)

        # Generate twice with top_k=1 (greedy)
        rng1 = jax.random.key(42)
        gen1 = generate(
            model,
            jax_params,
            tokens,
            max_new_tokens=5,
            temperature=1.0,
            top_k=1,
            rng_key=rng1,
        )

        rng2 = jax.random.key(123)
        gen2 = generate(
            model,
            jax_params,
            tokens,
            max_new_tokens=5,
            temperature=1.0,
            top_k=1,
            rng_key=rng2,
        )

        # Should be identical
        assert gen1 == gen2

    def test_generation_quality(self, gpt2_config, hf_tokenizer):
        """Test generation produces reasonable completions."""
        from pretrained.huggingface.loader import convert_hf_to_jax

        jax_params = convert_hf_to_jax("gpt2", gpt2_config)
        model = GPT(gpt2_config)

        test_prompts = [
            "Once upon a time",
            "The quick brown fox",
            "In conclusion,",
        ]

        for prompt in test_prompts:
            tokens = hf_tokenizer.encode(prompt)
            rng = jax.random.key(42)

            generated_tokens = generate(
                model,
                jax_params,
                tokens,
                max_new_tokens=15,
                temperature=0.7,
                top_k=40,
                rng_key=rng,
            )

            text = hf_tokenizer.decode(generated_tokens)

            # Basic quality checks
            assert len(text) > len(prompt)
            assert text.startswith(prompt)
            # Should have multiple words
            assert len(text.split()) > len(prompt.split())


class TestWeightMapping:
    """Test specific weight mapping from HF to JAX."""

    def test_embedding_weights_match(self, gpt2_config, hf_model):
        """Test embedding weights are correctly mapped."""
        from pretrained.huggingface.loader import convert_hf_to_jax

        jax_params = convert_hf_to_jax("gpt2", gpt2_config)

        # Get HF embeddings
        hf_wte = hf_model.transformer.wte.weight.detach().cpu().numpy()
        hf_wpe = hf_model.transformer.wpe.weight.detach().cpu().numpy()

        # Get JAX embeddings
        jax_wte = jax_params["Emb_0"]["Embed_0"]["embedding"]
        jax_wpe = jax_params["Emb_0"]["Embed_1"]["embedding"]

        # Should match (within precision)
        assert jnp.allclose(jax_wte, hf_wte, atol=1e-5)
        assert jnp.allclose(jax_wpe, hf_wpe, atol=1e-5)

    def test_attention_weights_shape(self, gpt2_config):
        """Test attention weights have correct shapes."""
        from pretrained.huggingface.loader import convert_hf_to_jax

        jax_params = convert_hf_to_jax("gpt2", gpt2_config)

        # Check first transformer block attention
        block0 = jax_params["TransformerBlock_0"]
        attn = block0["SelfAttention_0"]

        # Q, K, V projections
        assert attn["Wq"]["kernel"].shape == (768, 768)
        assert attn["Wk"]["kernel"].shape == (768, 768)
        assert attn["Wv"]["kernel"].shape == (768, 768)

        # Output projection
        assert attn["Wo"]["kernel"].shape == (768, 768)

    def test_mlp_weights_shape(self, gpt2_config):
        """Test MLP weights have correct shapes."""
        from pretrained.huggingface.loader import convert_hf_to_jax

        jax_params = convert_hf_to_jax("gpt2", gpt2_config)

        # Check first transformer block MLP
        block0 = jax_params["TransformerBlock_0"]
        mlp = block0["FFN_0"]

        # FF dim should be 4 * emb_dim = 3072
        assert mlp["Dense_0"]["kernel"].shape == (768, 3072)
        assert mlp["Dense_1"]["kernel"].shape == (3072, 768)


class TestEndToEndPipeline:
    """Test complete pipeline from HF to generation."""

    def test_full_pipeline(self, gpt2_config, hf_tokenizer):
        """Test complete pipeline: load weights -> generate text."""
        from pretrained.huggingface.loader import convert_hf_to_jax

        # 1. Load weights
        jax_params = convert_hf_to_jax("gpt2", gpt2_config)

        # 2. Create model
        model = GPT(gpt2_config)

        # 3. Tokenize prompt
        prompt = "Hello world, this is"
        tokens = hf_tokenizer.encode(prompt)

        # 4. Generate
        rng = jax.random.key(42)
        generated_tokens = generate(
            model,
            jax_params,
            tokens,
            max_new_tokens=20,
            temperature=0.8,
            top_k=50,
            rng_key=rng,
        )

        # 5. Decode
        text = hf_tokenizer.decode(generated_tokens)

        # Verify
        assert text.startswith(prompt)
        assert len(text) > len(prompt)
        print(f"\nGenerated: {text}")
