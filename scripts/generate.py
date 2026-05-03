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
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import numpy as np
import tiktoken

from janogpt import GPT, Config
from janogpt.inference import generate

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


def load_pretrained():
    """Load pretrained HuggingFace GPT2 weights."""
    from pretrained.huggingface.loader import load_hf_gpt2_weights

    # Config for HF GPT2
    config = Config(
        voc_size=50257,  # HF vocab size (not padded like our trained model)
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
    config = Config(**restored["config"])

    # Create model
    model = GPT(config)

    return model, restored["state"].params, config


def generate_streaming(model, params, prompt_tokens, max_new_tokens, temperature, top_k, rng_key, enc):
    """
    Generate tokens one at a time and yield each decoded token.

    Yields:
        tuple: (token_text, is_final, tokens_per_sec)
    """
    from janogpt.inference import sample_next_token

    # Start with prompt
    current_tokens = prompt_tokens
    start_time = time.time()

    for i in range(max_new_tokens):
        # Get logits for current sequence
        logits = model.apply({"params": params}, current_tokens[None, :], inference=True)
        next_token_logits = logits[0, -1, :]  # (voc_size,)

        # Sample next token
        rng_key, sample_key = jax.random.split(rng_key)
        next_token = sample_next_token(next_token_logits, temperature, top_k, sample_key)

        # Append to sequence
        current_tokens = jnp.append(current_tokens, next_token)

        # Decode just the new token
        token_text = enc.decode([int(next_token)])

        # Calculate tokens/sec
        elapsed = time.time() - start_time
        tokens_per_sec = (i + 1) / elapsed if elapsed > 0 else 0

        is_final = (i == max_new_tokens - 1)
        yield token_text, is_final, tokens_per_sec

    return current_tokens


def main():
    parser = argparse.ArgumentParser(description="Generate text with JanoGPT")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained HuggingFace GPT2")
    parser.add_argument("--prompt", type=str, default="Hello, I am", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--interactive", action="store_true", help="Interactive mode - enter prompts continuously"
    )
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
        # Print header
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}🤖 JanoGPT Interactive Mode{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.DIM}Model: seq_len={config.seq_len}, voc_size={config.voc_size}{Colors.RESET}")
        print(f"{Colors.DIM}Commands: 'quit', 'config', 'set <param> <value>'{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}\n")

        # Settings
        max_tokens = args.max_tokens
        temperature = args.temperature
        top_k = args.top_k
        seed = args.seed

        while True:
            try:
                # Prompt
                prompt = input(f"{Colors.BRIGHT_GREEN}> {Colors.RESET}").strip()

                if not prompt:
                    continue

                # Commands
                if prompt.lower() in ["quit", "exit", "q"]:
                    print(f"{Colors.YELLOW}Goodbye!{Colors.RESET}")
                    break

                if prompt.lower() == "config":
                    print(f"\n{Colors.CYAN}Current Settings:{Colors.RESET}")
                    print(f"  {Colors.BRIGHT_CYAN}max_tokens{Colors.RESET}   = {max_tokens}")
                    print(f"  {Colors.BRIGHT_CYAN}temperature{Colors.RESET}  = {temperature}")
                    print(f"  {Colors.BRIGHT_CYAN}top_k{Colors.RESET}        = {top_k}")
                    print(f"  {Colors.DIM}(model seq_len = {config.seq_len}){Colors.RESET}\n")
                    continue

                # Set command
                if prompt.lower().startswith("set "):
                    parts = prompt.split()
                    if len(parts) != 3:
                        print(f"{Colors.RED}Error: Usage: set <param> <value>{Colors.RESET}")
                        print(f"{Colors.DIM}Available params: max_tokens, temperature, top_k{Colors.RESET}\n")
                        continue

                    param = parts[1].lower()
                    try:
                        value = float(parts[2]) if '.' in parts[2] else int(parts[2])
                    except ValueError:
                        print(f"{Colors.RED}Error: Invalid value '{parts[2]}'{Colors.RESET}\n")
                        continue

                    if param in ["max_tokens", "max_output_tokens"]:
                        if value > config.seq_len:
                            print(
                                f"{Colors.RED}Error: max_tokens={value} exceeds model seq_len={config.seq_len}{Colors.RESET}"
                            )
                            print(f"{Colors.DIM}Maximum allowed: {config.seq_len}{Colors.RESET}\n")
                            continue
                        max_tokens = int(value)
                        print(f"{Colors.GREEN}✓ Set max_tokens = {max_tokens}{Colors.RESET}\n")
                    elif param == "temperature":
                        temperature = float(value)
                        print(f"{Colors.GREEN}✓ Set temperature = {temperature}{Colors.RESET}\n")
                    elif param == "top_k":
                        top_k = int(value)
                        print(f"{Colors.GREEN}✓ Set top_k = {top_k}{Colors.RESET}\n")
                    else:
                        print(f"{Colors.RED}Error: Unknown parameter '{param}'{Colors.RESET}")
                        print(f"{Colors.DIM}Available: max_tokens, temperature, top_k{Colors.RESET}\n")
                    continue

                # Generate with streaming
                prompt_tokens = np.array(enc.encode(prompt), dtype=np.int32)

                # Check if prompt + max_tokens exceeds seq_len
                if len(prompt_tokens) + max_tokens > config.seq_len:
                    print(
                        f"{Colors.YELLOW}Warning: prompt ({len(prompt_tokens)} tokens) + max_tokens ({max_tokens}) "
                        f"exceeds seq_len ({config.seq_len}){Colors.RESET}"
                    )
                    print(f"{Colors.DIM}Reducing max_tokens to {config.seq_len - len(prompt_tokens)}{Colors.RESET}\n")
                    max_tokens = config.seq_len - len(prompt_tokens)

                rng_key = jax.random.key(seed)
                seed += 1  # Change seed each time for variety

                print(f"{Colors.BRIGHT_MAGENTA}", end="", flush=True)

                # Stream generation
                tokens_generated = 0
                for token_text, is_final, tokens_per_sec in generate_streaming(
                    model, params, prompt_tokens, max_tokens, temperature, top_k, rng_key, enc
                ):
                    print(token_text, end="", flush=True)
                    tokens_generated += 1

                # Print stats
                print(f"{Colors.RESET}")
                print(
                    f"{Colors.DIM}[{tokens_generated} tokens, {tokens_per_sec:.1f} tok/s]{Colors.RESET}\n"
                )

            except (KeyboardInterrupt, EOFError):
                print(f"\n{Colors.YELLOW}Goodbye!{Colors.RESET}")
                break
            except Exception as e:
                print(f"\n{Colors.RED}Error: {e}{Colors.RESET}\n")
                continue

        return 0

    # Single prompt mode
    prompt_tokens = np.array(enc.encode(args.prompt), dtype=np.int32)
    print(f'\nPrompt: "{args.prompt}"')
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
        rng_key=rng_key,
    )

    # Decode
    text = enc.decode(generated.tolist())
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main() or 0)
