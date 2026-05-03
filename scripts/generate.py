#!/usr/bin/env python3
"""
Text generation script for JanoGPT.

Usage:
    # Interactive mode with pretrained GPT-2 (recommended!)
    python scripts/generate.py --pretrained --interactive

    # Single prompt with pretrained GPT-2
    python scripts/generate.py --pretrained --prompt "Once upon a time"

    # Interactive mode with your trained checkpoint
    python scripts/generate.py --checkpoint output/checkpoints/step_10000 --interactive

    # Single prompt with your checkpoint
    python scripts/generate.py --checkpoint output/checkpoints/step_10000 --prompt "Hello, I am"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import tiktoken

from janogpt import GPT, Config
from janogpt.inference import generate


def load_pretrained():
    """Load pretrained HuggingFace GPT2 weights."""
    from pretrained.huggingface.loader import load_hf_gpt2_weights

    # Config for HF GPT2
    config = Config(
        voc_size=50257,  # HF vocab size
        num_blocks=12,
        emb_dim=768,
        num_heads=12,
        seq_len=1024,
        dropout_prob=0.0,
    )

    model = GPT(config)
    params = load_hf_gpt2_weights(model, config)

    return model, params, config


def load_checkpoint(checkpoint_path: str):
    """Load model from checkpoint."""
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    restored = checkpointer.restore(checkpoint_path)

    # Reconstruct config
    config = Config(**restored['config'])

    # Create model
    model = GPT(config)

    return model, restored['state'].params, config


def main():
    parser = argparse.ArgumentParser(description='Generate text with JanoGPT')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint directory')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained HuggingFace GPT2')
    parser.add_argument('--prompt', type=str, default="Hello, I am",
                        help='Text prompt')
    parser.add_argument('--max_tokens', type=int, default=50,
                        help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode - enter prompts continuously')
    args = parser.parse_args()

    if not args.checkpoint and not args.pretrained:
        print("Error: Must specify either --checkpoint or --pretrained")
        return 1

    # Load model
    if args.pretrained:
        print("Loading pretrained HuggingFace GPT2...")
        model, params, config = load_pretrained()
    else:
        print(f"Loading checkpoint from {args.checkpoint}...")
        model, params, config = load_checkpoint(args.checkpoint)

    print("✓ Model loaded")

    # Get tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Interactive mode
    if args.interactive:
        print("\n" + "=" * 80)
        print("Interactive Mode - Type prompts and press Enter")
        print("Commands: 'quit' or 'exit' to exit, 'config' to see settings")
        print("=" * 80)

        seed = args.seed
        while True:
            try:
                prompt = input("\n> ").strip()

                if not prompt:
                    continue

                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                if prompt.lower() == 'config':
                    print(f"Settings: max_tokens={args.max_tokens}, "
                          f"temperature={args.temperature}, top_k={args.top_k}")
                    continue

                # Generate
                prompt_tokens = np.array(enc.encode(prompt), dtype=np.int32)
                rng_key = jax.random.key(seed)
                seed += 1  # Change seed each time for variety

                generated = generate(
                    model,
                    params,
                    prompt_tokens,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    rng_key=rng_key
                )

                # Decode and print
                text = enc.decode(generated.tolist())
                print("\n" + text + "\n")

            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

        return 0

    # Single prompt mode
    prompt_tokens = np.array(enc.encode(args.prompt), dtype=np.int32)
    print(f"\nPrompt: \"{args.prompt}\"")
    print(f"Generating {args.max_tokens} tokens with temperature={args.temperature}...")

    # Generate
    rng_key = jax.random.key(args.seed)
    generated = generate(
        model,
        params,
        prompt_tokens,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rng_key=rng_key
    )

    # Decode
    text = enc.decode(generated.tolist())
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


if __name__ == '__main__':
    sys.exit(main() or 0)
