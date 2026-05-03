# JanoGPT

A clean, educational GPT-2 implementation in JAX/Flax with production-quality training infrastructure.

**Focus:** Understand transformers deeply by building GPT-2 from scratch in JAX.

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

# Install optional dependencies
pip install -e ".[data,huggingface]"  # For dataset download and HF models
```

### Download Data

```bash
# Install kagglehub
pip install kagglehub

# Download OpenWebText dataset (~20GB)
python data/openwebtext/prepare.py
```

See [data/openwebtext/README.md](data/openwebtext/README.md) for details.

### Train

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

### Generate Text

```bash
# Generate from pretrained HuggingFace GPT-2
python scripts/generate.py \
  --pretrained \
  --prompt "Once upon a time" \
  --max_tokens 100 \
  --temperature 0.8

# Generate from your trained checkpoint
python scripts/generate.py \
  --checkpoint output_1gpu/checkpoints/step_10000 \
  --prompt "Hello, I am" \
  --max_tokens 50
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

### Sampling Parameters

```bash
python scripts/generate.py \
  --checkpoint output_1gpu/checkpoints/step_50000 \
  --prompt "Once upon a time" \
  --max_tokens 100 \          # Number of tokens to generate
  --temperature 0.8 \          # Higher = more random (0.7-1.0 typical)
  --top_k 50 \                # Top-k sampling (40-50 typical)
  --seed 42                   # Random seed for reproducibility
```

**Temperature:**
- `0.7` - More focused, coherent
- `0.8-0.9` - Balanced creativity
- `1.0+` - Very creative, possibly incoherent

**Top-k:**
- `40-50` - Standard setting
- Lower = more deterministic
- Higher = more diverse

### Using Pretrained Models

Load HuggingFace GPT-2 weights directly:

```bash
# Requires: pip install transformers torch
python scripts/generate.py \
  --pretrained \
  --prompt "The future of AI is" \
  --max_tokens 100
```

This downloads GPT-2 124M from HuggingFace and converts weights to JAX format automatically.

## Loading Pretrained Weights

### From HuggingFace

```python
from pretrained.huggingface.loader import load_hf_gpt2_weights
from janogpt import GPT, Config

# Create config matching HF GPT-2
config = Config(
    voc_size=50257,  # HF vocab size
    num_blocks=12,
    emb_dim=768,
    num_heads=12,
    seq_len=1024,
)

model = GPT(config)
params = load_hf_gpt2_weights(model, config)

# Now use params for generation
```

Supported models:
- `gpt2` (124M)
- `gpt2-medium` (355M)
- `gpt2-large` (774M)
- `gpt2-xl` (1.5B)

### From Checkpoint

```python
import orbax.checkpoint as ocp
from janogpt import GPT, Config

checkpointer = ocp.PyTreeCheckpointer()
restored = checkpointer.restore("output_1gpu/checkpoints/step_10000")

config = Config(**restored['config'])
model = GPT(config)
params = restored['state'].params

# Use for inference or resume training
```

## Testing

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
