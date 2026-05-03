"""
Inference and text generation utilities.
"""

import jax
import jax.numpy as jnp
import numpy as np


def generate(
    model,
    params,
    prompt_tokens: np.ndarray,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 50,
    rng_key = None
):
    """
    Generate tokens autoregressively with temperature sampling.

    Args:
        model: GPT model
        params: Model parameters
        prompt_tokens: Input tokens (1D array, no padding)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Only sample from top-k tokens (0 = no filtering)
        rng_key: Random key for sampling

    Returns:
        Generated tokens (including prompt)
    """
    if rng_key is None:
        rng_key = jax.random.key(42)

    # Start with prompt (no padding)
    generated = list(prompt_tokens)

    for _ in range(max_new_tokens):
        # Forward pass with current sequence
        input_batch = jnp.array([generated])  # (1, current_len)
        logits = model.apply(
            {'params': params},
            input_batch,
            inference=True
        )

        # Get logits for last position
        next_token_logits = logits[0, -1, :]  # (vocab_size,)

        # Apply temperature
        next_token_logits = next_token_logits / temperature

        # Apply top-k filtering
        if top_k > 0:
            top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, top_k)
            # Create mask for top-k
            logits_filtered = jnp.full_like(next_token_logits, -float('inf'))
            logits_filtered = logits_filtered.at[top_k_indices].set(top_k_logits)
            next_token_logits = logits_filtered

        # Sample from distribution
        rng_key, sample_key = jax.random.split(rng_key)
        next_token = jax.random.categorical(sample_key, next_token_logits)

        # Append to generated sequence
        generated.append(int(next_token))

        # Stop if we hit end of sequence token (50256 for GPT2)
        if int(next_token) == 50256:
            break

    return np.array(generated)


def get_top_k_predictions(
    model,
    params,
    input_tokens: np.ndarray,
    k: int = 10
) -> list:
    """
    Get top-k next token predictions.

    Args:
        model: GPT model
        params: Model parameters
        input_tokens: Input tokens (seq_len,)
        k: Number of top predictions to return

    Returns:
        List of (token_id, probability) tuples
    """
    # Add batch dimension
    input_batch = jnp.array(input_tokens[None, :])  # (1, seq_len)

    # Forward pass (inference mode)
    logits = model.apply(
        {'params': params},
        input_batch,
        inference=True
    )

    # Get logits for last position
    last_logits = logits[0, -1, :]  # (vocab_size,)

    # Convert to probabilities
    probs = jax.nn.softmax(last_logits)

    # Get top-k
    top_k_indices = jnp.argsort(probs)[-k:][::-1]
    top_k_probs = probs[top_k_indices]

    results = []
    for idx, prob in zip(top_k_indices, top_k_probs):
        results.append((int(idx), float(prob)))

    return results
