# JanoGPT

A clean, educational GPT-2 implementation in JAX/Flax with production-quality training infrastructure.

**Focus:** Understand transformers deeply by building GPT-2 from scratch in JAX.

## 🚀 Try It Now (No Training Required!)

```bash
# Install dependencies
pip install -e ".[huggingface]"

# Chat with GPT-2 interactively
python scripts/generate.py --pretrained --interactive
```

Type your prompts and watch GPT-2 complete them in real-time. No model training needed—uses pretrained weights from HuggingFace!

## Features

- ✅ **Pure JAX/Flax implementation** - Clean, functional code
- ✅ **Multi-GPU training** - Automatic data parallelism with JAX `pmap`
- ✅ **Checkpoint management** - Save/resume training with Orbax
- ✅ **Pretrained model loading** - Load HuggingFace GPT-2 weights
- ✅ **WandB integration** - Track metrics and visualizations
- ✅ **Flexible configuration** - JSON config system for reproducibility
- ✅ **Text generation** - Sample from trained or pretrained models

## Quick Start

### Installation

```bash
# Clone repository
git clone git@github.com:hhe0u0/janogpt.git
cd janogpt

# Install dependencies
pip install -e .

# For HuggingFace models (recommended to try immediately)
pip install -e ".[huggingface]"

# For training (optional - requires dataset download)
pip install -e ".[data]"
```

### Try It Immediately with Pretrained GPT-2

**No training required!** Load pretrained weights from HuggingFace and start generating text:

```bash
# Install HuggingFace dependencies
pip install -e ".[huggingface]"

# Interactive mode - chat with GPT-2
python scripts/generate.py --pretrained --interactive

# Single prompt mode
python scripts/generate.py \
  --pretrained \
  --prompt "The meaning of life is" \
  --max_tokens 100 \
  --temperature 0.8
```

**Interactive mode example:**
```
> Once upon a time
Once upon a time, there was a young girl who lived in a small village...

> The future of AI is
The future of AI is bright. Machine learning will revolutionize...

> config
Settings: max_tokens=50, temperature=0.8, top_k=50

> quit
Goodbye!
```

### Train Your Own Model

#### 1. Download Data

```bash
# Install kagglehub
pip install kagglehub

# Download OpenWebText dataset (~20GB)
python data/openwebtext/prepare.py
```

See [data/openwebtext/README.md](data/openwebtext/README.md) for details.

#### 2. Train

```bash
# Quick smoke test (< 1 minute)
python scripts/train.py --config configs/smoke_test.json

# Single GPU training (A100 recommended)
python scripts/train.py --config configs/train_1gpu_a100.json

# Multi-GPU training (4-8 GPUs)
python scripts/train.py --config configs/train_4gpu_a100.json

# Disable WandB logging
python scripts/train.py --config configs/train_1gpu_a100.json --no_wandb
```

#### 3. Generate from Your Checkpoint

```bash
# Single prompt
python scripts/generate.py \
  --checkpoint output_1gpu/checkpoints/step_10000 \
  --prompt "Hello, I am" \
  --max_tokens 50

# Interactive mode
python scripts/generate.py \
  --checkpoint output_1gpu/checkpoints/step_10000 \
  --interactive
```

### Resume Training

```bash
# Training automatically saves checkpoints
# Resume from checkpoint:
python scripts/train.py \
  --config configs/train_1gpu_a100.json \
  --resume output_1gpu/checkpoints/step_10000
```

## Training Guide

### Configuration System

Training is controlled by JSON config files in `configs/`:

- `smoke_test.json` - Quick test (< 1 min, 10 steps, tiny model)
- `train_1gpu_a100.json` - Single GPU training (100K steps, ~2-3 days)
- `train_4gpu_a100.json` - Multi-GPU training (600K steps, ~1 week)

**Config structure:**

```json
{
  "model": {
    "num_blocks": 12,
    "emb_dim": 768,
    "num_heads": 12,
    "seq_len": 1024,
    "voc_size": 50304
  },
  "optimizer": {
    "learning_rate": 6e-4,
    "warmup_steps": 2000
  },
  "training": {
    "max_steps": 100000,
    "micro_batch_size": 4,
    "gradient_accumulation_steps": 128
  },
  "checkpointing": {
    "save_interval": 2000,
    "output_dir": "output_1gpu",
    "resume_from_checkpoint": null
  }
}
```

See [configs/](configs/) directory for complete examples.

### Hardware Requirements

**Minimum (CPU testing):**
- 16GB RAM
- Use `smoke_test.json`
- ~10 steps in 1 minute

**Recommended (Single GPU):**
- NVIDIA A100 (40GB) or V100 (32GB)
- Use `train_1gpu_a100.json`
- ~100K steps in 2-3 days
- ~0.5M tokens per step

**Production (Multi-GPU):**
- 4-8× A100 (80GB each)
- Use `train_4gpu_a100.json`
- ~600K steps in 1 week
- ~0.5M tokens per step

### Effective Batch Size

The effective batch size (tokens per gradient update) is:

```
effective_tokens = micro_batch_size × gradient_accumulation_steps × num_gpus × seq_len
```

**Example (1 GPU):**
```
4 × 128 × 1 × 1024 = 524,288 tokens ≈ 0.5M tokens/step
```

**Example (8 GPUs):**
```
4 × 16 × 8 × 1024 = 524,288 tokens ≈ 0.5M tokens/step
```

Target ~0.5M tokens per step for GPT-2 124M model (following Chinchilla scaling laws).

### Checkpointing

**Automatic saving:**
- Saves every `save_interval` steps (configured in JSON)
- Saves to `{output_dir}/checkpoints/step_{step}`
- Each checkpoint includes:
  - Model weights (`state.params`)
  - Optimizer state
  - Training config
  - RNG state

**Resume training:**

Option 1 - Set in config JSON:
```json
{
  "checkpointing": {
    "resume_from_checkpoint": "output_1gpu/checkpoints/step_10000"
  }
}
```

Option 2 - Override via CLI:
```bash
python scripts/train.py \
  --config configs/train_1gpu_a100.json \
  --resume output_1gpu/checkpoints/step_10000
```

**What gets restored:**
- ✅ Model weights
- ✅ Optimizer state (momentum, etc.)
- ✅ Training step count
- ✅ RNG state (for reproducibility)
- ✅ Learning rate schedule

Training continues from the exact state it was saved.

### Monitoring with WandB

Enable in config:
```json
{
  "wandb": {
    "enabled": true,
    "project": "janogpt",
    "run_name": "my-experiment",
    "tags": ["gpt2", "openwebtext"]
  }
}
```

**Tracked metrics:**
- Training loss (every log_interval steps)
- Validation loss (every eval_interval steps)
- Learning rate schedule
- Tokens per second
- GPU memory usage

Disable WandB: `python scripts/train.py --config ... --no_wandb`

## Text Generation

### Interactive Mode (Recommended)

**The easiest way to use the model!** Chat with GPT-2 in real-time:

```bash
# With pretrained HuggingFace GPT-2 (no training needed!)
python scripts/generate.py --pretrained --interactive

# With your trained checkpoint
python scripts/generate.py \
  --checkpoint output_1gpu/checkpoints/step_10000 \
  --interactive

# Adjust creativity
python scripts/generate.py \
  --pretrained \
  --interactive \
  --temperature 0.9 \    # Higher = more creative
  --max_tokens 100       # Longer completions
```

**Interactive features:**
- 🌈 **Colored output** - Beautiful CLI interface with syntax highlighting
- 🚀 **Streaming generation** - See tokens appear in real-time
- 📊 **Performance metrics** - Live tokens/sec display
- ⚙️ **Dynamic settings** - Adjust parameters without restarting

**Interactive commands:**
- Type any prompt and press Enter to generate
- `config` - Show current settings
- `set <param> <value>` - Change generation parameters
  - `set max_tokens 100` - Set max output length
  - `set temperature 0.9` - Adjust creativity
  - `set top_k 40` - Adjust diversity
- `quit` or `exit` - Exit interactive mode
- Ctrl+C - Exit

**Example session:**
```
================================================================================
🤖 JanoGPT Interactive Mode
================================================================================
Model: seq_len=1024, voc_size=50257
Commands: 'quit', 'config', 'set <param> <value>'
================================================================================

> The meaning of life is
The meaning of life is to find happiness and purpose in our daily experiences...
[25 tokens, 12.3 tok/s]

> set max_tokens 100
✓ Set max_tokens = 100

> set temperature 0.9
✓ Set temperature = 0.9

> Write a haiku about AI
Write a haiku about AI:
Silicon minds dream
Learning patterns from the world
Human thoughts reborn
[18 tokens, 15.7 tok/s]

> config
Current Settings:
  max_tokens   = 100
  temperature  = 0.9
  top_k        = 50
  (model seq_len = 1024)

> quit
Goodbye!
```

**Error handling:**
- Automatic validation of `max_tokens` against model's `seq_len`
- Warning if prompt + max_tokens exceeds context window
- Clear error messages with suggestions

### Single Prompt Mode

Generate text from a single prompt:

```bash
# With pretrained HuggingFace GPT-2
python scripts/generate.py \
  --pretrained \
  --prompt "Once upon a time" \
  --max_tokens 100 \
  --temperature 0.8

# With your trained checkpoint
python scripts/generate.py \
  --checkpoint output_1gpu/checkpoints/step_50000 \
  --prompt "The future of AI is" \
  --max_tokens 100
```

### Sampling Parameters

Control the generation style:

```bash
python scripts/generate.py \
  --pretrained \
  --prompt "Once upon a time" \
  --max_tokens 100 \          # Number of tokens to generate
  --temperature 0.8 \          # Higher = more random (0.7-1.0 typical)
  --top_k 50 \                # Top-k sampling (40-50 typical)
  --seed 42                   # Random seed for reproducibility
```

**Temperature:**
- `0.7` - More focused, coherent, deterministic
- `0.8-0.9` - Balanced creativity (recommended)
- `1.0+` - Very creative, possibly incoherent
- Lower = safer, more repetitive
- Higher = riskier, more diverse

**Top-k:**
- `40-50` - Standard setting (recommended)
- `20-30` - More focused, less diverse
- `100+` - More diverse, potentially incoherent
- Limits sampling to top-k most likely tokens

**Max tokens:**
- `50` - Short completion (1-2 sentences)
- `100` - Medium paragraph
- `200+` - Long-form generation

### Using Pretrained HuggingFace Models

Load GPT-2 weights from HuggingFace and use immediately (no training required):

```bash
# Install dependencies
pip install transformers torch

# Generate text with pretrained GPT-2
python scripts/generate.py \
  --pretrained \
  --prompt "The future of AI is" \
  --max_tokens 100

# Interactive mode (recommended!)
python scripts/generate.py --pretrained --interactive
```

**What happens:**
1. Downloads GPT-2 124M from HuggingFace (first time only, ~500MB)
2. Automatically converts PyTorch weights to JAX format
3. Ready to generate text immediately
4. Weights cached locally for future use

**Supported models:**
- `gpt2` (124M parameters) - Default, good for most uses
- `gpt2-medium` (355M) - Better quality, slower
- `gpt2-large` (774M) - High quality, requires more memory
- `gpt2-xl` (1.5B) - Best quality, requires 16GB+ GPU

To use larger models, modify `pretrained/huggingface/loader.py` and change the model name.

## Loading Pretrained Weights

### From HuggingFace (Easiest)

**Command-line (recommended):**

```bash
# Interactive mode - just chat with GPT-2!
python scripts/generate.py --pretrained --interactive

# Or single prompt
python scripts/generate.py \
  --pretrained \
  --prompt "Complete this sentence: The key to success is" \
  --max_tokens 50
```

**Python API:**

```python
from pretrained.huggingface.loader import load_hf_gpt2_weights
from janogpt import GPT, Config
from janogpt.inference import generate
import tiktoken
import jax
import numpy as np

# Create config matching HF GPT-2
config = Config(
    voc_size=50257,  # HF vocab size (different from trained model!)
    num_blocks=12,
    emb_dim=768,
    num_heads=12,
    seq_len=1024,
)

# Load pretrained weights from HuggingFace
model = GPT(config)
params = load_hf_gpt2_weights(model, config)

# Tokenize your prompt
enc = tiktoken.get_encoding("gpt2")
prompt = "The meaning of life is"
prompt_tokens = np.array(enc.encode(prompt), dtype=np.int32)

# Generate text
rng_key = jax.random.key(42)
generated = generate(
    model,
    params,
    prompt_tokens,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    rng_key=rng_key
)

# Decode and print
text = enc.decode(generated.tolist())
print(text)
```

**Supported HuggingFace models:**
- `gpt2` (124M) - Default, fast, good quality
- `gpt2-medium` (355M) - Better quality, slower
- `gpt2-large` (774M) - High quality, requires more memory
- `gpt2-xl` (1.5B) - Best quality, requires 16GB+ GPU

To use larger models, modify `pretrained/huggingface/loader.py`:
```python
# Change this line
model_name = "gpt2-medium"  # or "gpt2-large", "gpt2-xl"
```

### From Your Trained Checkpoint

**Command-line:**

```bash
# Interactive mode
python scripts/generate.py \
  --checkpoint output_1gpu/checkpoints/step_10000 \
  --interactive

# Single prompt
python scripts/generate.py \
  --checkpoint output_1gpu/checkpoints/step_50000 \
  --prompt "Hello, I am" \
  --max_tokens 50
```

**Python API:**

```python
import orbax.checkpoint as ocp
from janogpt import GPT, Config
from janogpt.inference import generate
import tiktoken
import jax
import numpy as np

# Load checkpoint
checkpointer = ocp.PyTreeCheckpointer()
restored = checkpointer.restore("output_1gpu/checkpoints/step_10000")

# Reconstruct model
config = Config(**restored['config'])
model = GPT(config)
params = restored['state'].params

# Generate text
enc = tiktoken.get_encoding("gpt2")
prompt_tokens = np.array(enc.encode("Hello, I am"), dtype=np.int32)
rng_key = jax.random.key(42)

generated = generate(
    model,
    params,
    prompt_tokens,
    max_new_tokens=50,
    temperature=0.8,
    top_k=50,
    rng_key=rng_key
)

text = enc.decode(generated.tolist())
print(text)
```

**What's in a checkpoint:**
- Model weights (`state.params`)
- Optimizer state (Adam momentum, variance)
- Training step count
- RNG state
- Full training config

You can resume training or just use the weights for inference.

## Development & Code Quality

### Static Analysis

Install development dependencies:
```bash
pip install -e ".[dev]"
pre-commit install  # Setup git hooks
```

**Run checks manually:**
```bash
# Format code (auto-fix)
make format

# Lint code (report issues)
make lint

# Type checking
make typecheck

# Security checks
make security

# Run all checks
make check

# Run tests
make test

# Everything (format + check + test)
make all
```

**Pre-commit hooks** (automatic on git commit):
- Ruff formatting and linting
- MyPy type checking
- Bandit security scanning
- Trailing whitespace, large files, etc.

**CI/CD:** GitHub Actions runs checks on all PRs:
- Lint & format verification
- Type checking (mypy)
- Security scanning (bandit)
- Tests on Python 3.9, 3.10, 3.11

### Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/unit/test_model.py

# Run with coverage
pytest --cov=janogpt --cov-report=html

# Quick smoke test
python scripts/train.py --config configs/smoke_test.json
```

Test structure:
- `tests/unit/` - Unit tests for individual components
- `tests/integration/` - End-to-end training tests
- `configs/smoke_test.json` - Fast integration test config

## Project Structure

```
janogpt/
├── janogpt/              # Main package
│   ├── __init__.py
│   ├── model.py          # GPT model (transformer blocks, attention)
│   ├── trainer.py        # Training loop, optimization, checkpointing
│   ├── inference.py      # Text generation, sampling strategies
│   ├── config.py         # Configuration dataclass and loading
│   ├── protocols.py      # Type protocols for duck typing
│   ├── logger.py         # Logging (console, WandB)
│   └── utils.py          # Data loaders, utilities
│
├── scripts/
│   ├── train.py          # Training entry point
│   └── generate.py       # Text generation entry point
│
├── configs/              # Training configurations
│   ├── smoke_test.json
│   ├── train_1gpu_a100.json
│   └── train_4gpu_a100.json
│
├── data/
│   └── openwebtext/      # Dataset directory
│       ├── prepare.py    # Download script
│       └── README.md
│
├── pretrained/
│   └── huggingface/      # HuggingFace model loading
│       ├── __init__.py
│       └── loader.py
│
├── tests/
│   ├── unit/             # Component tests
│   └── integration/      # End-to-end tests
│
├── notebooks/            # Jupyter notebooks for exploration
├── pyproject.toml        # Package configuration
└── README.md
```

## Model Architecture

**GPT-2 124M configuration:**
- 12 transformer blocks
- 768 embedding dimension
- 12 attention heads (64 dim per head)
- 1024 context length
- 50,304 vocabulary size (padded for efficiency)
- ~124M parameters

**Key components:**
- `CausalSelfAttention` - Multi-head attention with causal masking
- `MLP` - Feed-forward network (4× hidden dim, GELU activation)
- `Block` - Transformer block (attention + MLP with residual)
- `GPT` - Full model (embedding + blocks + LM head)

See [janogpt/model.py](janogpt/model.py) for implementation.

## Training Details

**Optimizer:** AdamW
- Learning rate: 6e-4 (peak)
- Warmup: 2,000 steps (linear)
- Decay: Cosine to 6e-5 (10% of peak)
- Weight decay: 0.1
- Beta1: 0.9, Beta2: 0.95
- Gradient clipping: 1.0

**Regularization:**
- Dropout: 0.1 (embedding, attention, residual)
- Weight decay: 0.1 (except biases and LayerNorm)

**Data:**
- OpenWebText dataset (~9B tokens)
- GPT-2 BPE tokenizer (50,257 vocab)
- 1024 token context windows

**Batch size:**
- Target: 0.5M tokens per gradient update
- Achieved via gradient accumulation and multi-GPU

## Performance Benchmarks

**Single A100 (40GB):**
- ~1,000 tokens/sec
- ~2 steps/sec (with 0.5M token batches)
- 100K steps in 2-3 days

**8× A100 (80GB):**
- ~8,000 tokens/sec
- ~16 steps/sec (with 0.5M token batches)
- 600K steps in 1 week

**Memory usage:**
- Model: ~500MB
- Optimizer state: ~1GB
- Activations: Depends on micro_batch_size and seq_len
- Total: ~20-30GB per GPU for typical training

## Common Issues

### Out of Memory (OOM)

Reduce `micro_batch_size` in config:
```json
{
  "training": {
    "micro_batch_size": 2,  // Reduce from 4
    "gradient_accumulation_steps": 256  // Increase to maintain effective batch
  }
}
```

### Slow Training

Check effective batch size:
```python
# Should be ~0.5M tokens per step
effective_tokens = micro_batch * grad_accum * num_gpus * seq_len
```

Enable GPU:
```bash
# Check JAX sees GPUs
python -c "import jax; print(jax.devices())"
```

### Checkpoint Loading Fails

Ensure config matches checkpoint:
```bash
# Config is saved in checkpoint
# Restore will fail if model architecture differs
```

## Contributing

This is an educational project. Contributions welcome!

**Focus areas:**
- Improving code clarity and documentation
- Adding more examples and notebooks
- Performance optimizations
- Better error messages

## License

MIT License - See LICENSE file

## Acknowledgments

- **Architecture:** Based on GPT-2 by OpenAI (Radford et al., 2019)
- **Framework:** JAX/Flax by Google
- **Dataset:** OpenWebText (pre-tokenized by windmaple)
- **Inspiration:** Karpathy's nanoGPT and micrograd

## References

- [Language Models are Unsupervised Multitask Learners (GPT-2 paper)](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Attention Is All You Need (Transformer paper)](https://arxiv.org/abs/1706.03762)
- [JAX Documentation](https://jax.readthedocs.io/)
- [Flax Documentation](https://flax.readthedocs.io/)
