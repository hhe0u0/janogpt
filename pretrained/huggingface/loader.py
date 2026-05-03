"""
Convert HuggingFace GPT2 weights to our JAX/Flax format.

HuggingFace format (PyTorch):
- transformer.wte.weight: (vocab_size, emb_dim)
- transformer.wpe.weight: (seq_len, emb_dim)
- transformer.h[i].ln_1.weight/bias: Layer norm before attention
- transformer.h[i].attn.c_attn.weight/bias: QKV projection (emb_dim, 3*emb_dim)
- transformer.h[i].attn.c_proj.weight/bias: Output projection (emb_dim, emb_dim)
- transformer.h[i].ln_2.weight/bias: Layer norm before MLP
- transformer.h[i].mlp.c_fc.weight/bias: MLP first layer (emb_dim, 4*emb_dim)
- transformer.h[i].mlp.c_proj.weight/bias: MLP second layer (4*emb_dim, emb_dim)
- transformer.ln_f.weight/bias: Final layer norm

Our JAX/Flax format:
- Emb/Embed_0/embedding: Token embeddings (vocab_size, emb_dim)
- Emb/Embed_1/embedding: Position embeddings (seq_len, emb_dim)
- AttnBlock_i/LayerNorm_0/scale: Pre-attention layer norm
- AttnBlock_i/MultiHeadDotProductAttention_0/...: Attention
- AttnBlock_i/LayerNorm_1/scale: Pre-MLP layer norm
- AttnBlock_i/Dense_0/kernel: MLP first layer
- AttnBlock_i/Dense_1/kernel: MLP second layer
- LayerNorm_0/scale: Final layer norm
"""

import jax
import jax.numpy as jnp
from flax.core import freeze


def load_hf_gpt2_weights(our_model, config, model_name="gpt2"):
    """
    Load pretrained GPT-2 weights from HuggingFace and convert to JAX format.

    Args:
        our_model: Our Flax GPT model
        config: Our model config
        model_name: HuggingFace model name (gpt2, gpt2-medium, gpt2-large, gpt2-xl)

    Returns:
        JAX parameters in our model's format
    """
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        raise ImportError(
            "transformers not installed. Run: pip install transformers torch"
        )

    print(f"Downloading {model_name} from HuggingFace...")
    hf_model = GPT2LMHeadModel.from_pretrained(model_name)
    print("✓ Download complete")

    return convert_hf_to_jax(hf_model, our_model, config)


def convert_hf_to_jax(hf_model, our_model, config):
    """
    Convert HuggingFace GPT2 weights to our JAX format.

    Args:
        hf_model: HuggingFace GPT2LMHeadModel
        our_model: Our Flax GPT model
        config: Our model config

    Returns:
        JAX parameters in our model's format
    """
    print("Converting HuggingFace weights to JAX format...")

    hf_state = hf_model.state_dict()

    # Initialize our model to get structure
    rng = jax.random.key(42)
    dummy_input = jnp.zeros((1, config.seq_len), dtype=jnp.uint16)
    variables = our_model.init({"params": rng, "dropout": rng}, dummy_input, inference=True)

    # We'll build params as a mutable dict - need to unfreeze properly
    from flax.core import unfreeze

    params = unfreeze(variables["params"])

    # Convert embeddings
    print("  Converting embeddings...")

    # Token embeddings: wte.weight -> Emb_0/Embed_0/embedding
    hf_wte = hf_state["transformer.wte.weight"].cpu().numpy()  # (vocab, emb)
    if "Emb_0" not in params:
        params["Emb_0"] = {}
    params["Emb_0"]["Embed_0"] = {"embedding": jnp.array(hf_wte)}

    # Position embeddings: wpe.weight -> Emb_0/Embed_1/embedding
    hf_wpe = hf_state["transformer.wpe.weight"].cpu().numpy()  # (seq, emb)
    params["Emb_0"]["Embed_1"] = {"embedding": jnp.array(hf_wpe)}

    print(f"    Token embeddings: {hf_wte.shape} -> Emb_0/Embed_0")
    print(f"    Position embeddings: {hf_wpe.shape} -> Emb_0/Embed_1")

    # Convert transformer blocks
    for i in range(config.num_blocks):
        print(f"  Converting block {i}...")

        block_key = f"AttnBlock_{i}"
        if block_key not in params:
            params[block_key] = {}

        # Pre-attention LayerNorm
        ln1_weight = hf_state[f"transformer.h.{i}.ln_1.weight"].cpu().numpy()
        ln1_bias = hf_state[f"transformer.h.{i}.ln_1.bias"].cpu().numpy()
        params[block_key]["LayerNorm_0"] = {
            "scale": jnp.array(ln1_weight),
            "bias": jnp.array(ln1_bias),
        }

        # Attention weights
        # HF uses Conv1D format: stored as (in_features, out_features) = (emb, 3*emb)
        # c_attn contains Q, K, V projections concatenated
        c_attn_weight = (
            hf_state[f"transformer.h.{i}.attn.c_attn.weight"].cpu().numpy()
        )  # (emb, 3*emb) = (768, 2304)
        c_attn_bias = (
            hf_state[f"transformer.h.{i}.attn.c_attn.bias"].cpu().numpy()
        )  # (3*emb,) = (2304,)

        # Split into Q, K, V
        qkv_weight = c_attn_weight  # (emb, 3*emb)
        qkv_bias = c_attn_bias  # (3*emb,)

        # Flax MultiHeadDotProductAttention expects:
        # - query/key/value kernels: (emb, num_heads, head_dim)
        # - query/key/value bias: (num_heads, head_dim)

        emb_dim = config.emb_dim
        num_heads = config.num_heads
        head_dim = emb_dim // num_heads

        # Split QKV
        q_weight = qkv_weight[:, :emb_dim]  # (emb, emb)
        k_weight = qkv_weight[:, emb_dim : 2 * emb_dim]
        v_weight = qkv_weight[:, 2 * emb_dim :]

        q_bias = qkv_bias[:emb_dim]
        k_bias = qkv_bias[emb_dim : 2 * emb_dim]
        v_bias = qkv_bias[2 * emb_dim :]

        # Reshape to (emb, num_heads, head_dim)
        q_kernel = q_weight.reshape(emb_dim, num_heads, head_dim)
        k_kernel = k_weight.reshape(emb_dim, num_heads, head_dim)
        v_kernel = v_weight.reshape(emb_dim, num_heads, head_dim)

        q_bias_reshaped = q_bias.reshape(num_heads, head_dim)
        k_bias_reshaped = k_bias.reshape(num_heads, head_dim)
        v_bias_reshaped = v_bias.reshape(num_heads, head_dim)

        # Output projection (emb, emb) in Conv1D format
        c_proj_weight = (
            hf_state[f"transformer.h.{i}.attn.c_proj.weight"].cpu().numpy()
        )  # (emb, emb) = (768, 768)
        c_proj_bias = hf_state[f"transformer.h.{i}.attn.c_proj.bias"].cpu().numpy()  # (emb,)

        # Reshape output projection to (num_heads, head_dim, emb)
        # Input to out projection is (batch, seq, num_heads, head_dim) → needs (num_heads, head_dim, emb)
        out_kernel = c_proj_weight.reshape(num_heads, head_dim, emb_dim)

        params[block_key]["MultiHeadDotProductAttention_0"] = {
            "query": {"kernel": jnp.array(q_kernel), "bias": jnp.array(q_bias_reshaped)},
            "key": {"kernel": jnp.array(k_kernel), "bias": jnp.array(k_bias_reshaped)},
            "value": {"kernel": jnp.array(v_kernel), "bias": jnp.array(v_bias_reshaped)},
            "out": {"kernel": jnp.array(out_kernel), "bias": jnp.array(c_proj_bias)},
        }

        # Pre-MLP LayerNorm
        ln2_weight = hf_state[f"transformer.h.{i}.ln_2.weight"].cpu().numpy()
        ln2_bias = hf_state[f"transformer.h.{i}.ln_2.bias"].cpu().numpy()
        params[block_key]["LayerNorm_1"] = {
            "scale": jnp.array(ln2_weight),
            "bias": jnp.array(ln2_bias),
        }

        # MLP
        # First layer (emb -> 4*emb) in Conv1D format
        mlp_fc_weight = (
            hf_state[f"transformer.h.{i}.mlp.c_fc.weight"].cpu().numpy()
        )  # (emb, 4*emb) = (768, 3072)
        mlp_fc_bias = hf_state[f"transformer.h.{i}.mlp.c_fc.bias"].cpu().numpy()  # (4*emb,)

        params[block_key]["Dense_0"] = {
            "kernel": jnp.array(mlp_fc_weight),
            "bias": jnp.array(mlp_fc_bias),
        }

        # Second layer (4*emb -> emb) in Conv1D format
        mlp_proj_weight = (
            hf_state[f"transformer.h.{i}.mlp.c_proj.weight"].cpu().numpy()
        )  # (4*emb, emb) = (3072, 768)
        mlp_proj_bias = hf_state[f"transformer.h.{i}.mlp.c_proj.bias"].cpu().numpy()  # (emb,)

        params[block_key]["Dense_1"] = {
            "kernel": jnp.array(mlp_proj_weight),
            "bias": jnp.array(mlp_proj_bias),
        }

    # Final LayerNorm
    print("  Converting final LayerNorm...")
    ln_f_weight = hf_state["transformer.ln_f.weight"].cpu().numpy()
    ln_f_bias = hf_state["transformer.ln_f.bias"].cpu().numpy()
    params["LayerNorm_0"] = {"scale": jnp.array(ln_f_weight), "bias": jnp.array(ln_f_bias)}

    print("✓ Weight conversion complete!")

    return freeze(params)
