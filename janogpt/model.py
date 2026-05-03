# standard libraries
import os
import numpy as np
import math
import json
from functools import partial
from dataclasses import dataclass

# tqdm for loading bars
from tqdm.auto import tqdm

# tokenizer
import tiktoken

# observability
import wandb

# JAX
import jax
import jax.numpy as jnp
from jax import random
from flax.training.train_state import TrainState
import flax.linen as nn
import optax, orbax
import orbax.checkpoint as ocp

## JAX mesh
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

#
from .config import Config

def make_causal_mask(seq_len, dtype=jnp.float32):
    """
    Create causal mask for attention.

    Args:
        seq_len: Sequence length
        dtype: Data type

    Returns:
        Mask of shape (1, 1, seq_len, seq_len)
    """
    # shape: (1, 1, seq_len, seq_len) — broadcast over batch and heads
    idx = jnp.arange(seq_len)
    mask = idx[None, :] <= idx[:, None]   # lower-triangular, True where attend is allowed
    mask = mask[None, None, :, :]         # (1, 1, T, T)
    return mask.astype(dtype)

def count_params(params):
    p = jax.tree_util.tree_map(lambda a : a.size if isinstance(a, jnp.ndarray) else 0, params)
    return jax.tree_util.tree_reduce(lambda a,b : a+b, p)

class AttnBlock(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x, mask=None, inference=False):

        mlp = nn.Sequential([
            nn.Dense(features=self.config.ff_dim),
            nn.gelu,
            nn.Dropout(self.config.dropout_prob, deterministic=inference),
            nn.Dense(features=self.config.emb_dim),
            nn.Dropout(self.config.dropout_prob, deterministic=inference),
        ])

        # Pre-LN Transformer Block (GPT-2 architecture)
        # Attention path with residual
        x_norm = nn.LayerNorm(
            epsilon=self.config.epsilon,
            dtype=self.config.dtype,
        )(x)
        x_attn = nn.MultiHeadDotProductAttention(
            num_heads=self.config.num_heads,
            qkv_features = self.config.emb_dim,
            out_features = self.config.emb_dim,
            dropout_rate = self.config.dropout_prob,
            deterministic=inference,
            dtype=self.config.dtype,
        )(x_norm, mask=mask)  # Pass causal mask here
        x = x + x_attn  # Residual connection

        # MLP path with residual
        x_norm = nn.LayerNorm(epsilon=self.config.epsilon)(x)
        x_mlp = mlp(x_norm)
        x = x + x_mlp  # Residual connection

        return x

class Emb(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x, inference=False):
        B, T = x.shape
        tkn = x
        pos = jnp.arange(0, T)[None]  # Position indices: (1, T)
        # IMPORTANT: Create wte before wpe to match HuggingFace naming
        # Embed_0 = wte (token embeddings), Embed_1 = wpe (position embeddings)
        wte = nn.Embed(self.config.vocab_size, self.config.emb_dim)  # Token embeddings
        wpe = nn.Embed(self.config.seq_len, self.config.emb_dim)  # Position embeddings
        tkn_emb = wte(tkn)
        pos_emb = wpe(pos)
        # Add dropout after embedding sum (standard GPT-2 practice)
        x = nn.Dropout(self.config.dropout_prob, deterministic=inference)(pos_emb + tkn_emb)
        return (wte, x)

class GPT(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, x, inference=False):
        emb = Emb(self.config)

        # Create causal mask dynamically based on actual sequence length
        seq_len = x.shape[1]
        mask = make_causal_mask(seq_len, dtype=self.config.dtype)

        wte, x = emb(x, inference=inference)
        for _ in range(self.config.num_blocks):
            x = AttnBlock(self.config)(x, mask, inference)
        x = nn.LayerNorm(epsilon=self.config.epsilon)(x)
        return wte.attend(x)

if __name__ == '__main__':
    config = Config()
    x = jnp.zeros((1, config.seq_len), dtype=jnp.uint16)
    model = GPT(config)
    rng = random.key(3407)
    rng, key_x, key_params, key_dropout_init, key_dropout_apply = random.split(rng, 5)
    params = model.init({'params': key_params, 'dropout': key_dropout_init}, x)['params']
    binded_model = model.bind({'params': params}, rngs={'dropout': key_dropout_apply})
    out = binded_model(x, inference=False)
    print(f'out.shape={out.shape}')
    # out.shape=(1, 1024, 50304)
    # count_params=124.49M
    print(f"count_params={count_params(params) / 1e6:.2f}M")
