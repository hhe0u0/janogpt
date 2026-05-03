"""
JanoGPT: JAX implementation of GPT-2.

A clean, educational implementation of GPT-2 in JAX/Flax with:
- Full GPT-2 architecture (124M parameters)
- Compatible with HuggingFace pretrained weights
- Single and multi-device training (CPU/GPU/TPU)
- Gradient accumulation
- Modern training features (WandB logging, checkpointing)
"""

__version__ = "0.1.0"

from janogpt.model import GPT, count_params
from janogpt.config import Config
from janogpt.trainer import Trainer

__all__ = [
    "GPT",
    "Config",
    "Trainer",
    "count_params",
]
