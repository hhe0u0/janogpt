"""
Base interfaces for dependency injection in janogpt trainer.

All components implement these interfaces for easy testing and swapping.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Dict

import jax.numpy as jnp

# ========== Data Loading ==========


class DataLoader(ABC):
    """Base interface for data loaders."""

    @abstractmethod
    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        """
        Yield batches of data.

        Returns:
            Iterator yielding dicts with 'input_ids' of shape (batch_size, seq_len)
        """
        pass

    @abstractmethod
    def get_batch(self) -> Dict[str, jnp.ndarray]:
        """Get a single batch."""
        pass


# ========== Model ==========


class Model(ABC):
    """Base interface for models."""

    @abstractmethod
    def init(self, rng, input_shape) -> Dict[str, Any]:
        """
        Initialize model parameters.

        Args:
            rng: Random key
            input_shape: Shape of input (batch_size, seq_len)

        Returns:
            Dict with 'params' key containing parameters
        """
        pass

    @abstractmethod
    def apply(self, variables, inputs, **kwargs):
        """
        Apply model to inputs.

        Args:
            variables: Dict with 'params' and other state
            inputs: Input tensor
            **kwargs: Additional arguments (e.g., rngs, inference mode)

        Returns:
            Model outputs (logits for language models)
        """
        pass

    @abstractmethod
    def count_params(self, params) -> int:
        """Count number of parameters."""
        pass


# ========== Evaluation ==========


class Evaluator(ABC):
    """Base interface for evaluators."""

    @abstractmethod
    def evaluate(
        self,
        state,
        compute_loss_fn,
        step: int,
    ) -> Dict[str, float]:
        """
        Run evaluation.

        Args:
            state: Training state with params
            compute_loss_fn: Function to compute loss given (params, batch, rng, training)
            step: Current training step

        Returns:
            Dict of metric_name -> value
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Evaluator name for logging."""
        pass


# ========== Logging ==========


class Logger(ABC):
    """Base interface for loggers."""

    @abstractmethod
    def init(self, config: Any, model_params: int):
        """
        Initialize logger.

        Args:
            config: Training configuration
            model_params: Number of model parameters
        """
        pass

    @abstractmethod
    def log(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics.

        Args:
            metrics: Dict of metric_name -> value
            step: Current training step
        """
        pass

    @abstractmethod
    def finish(self):
        """Finalize logging (e.g., close wandb run)."""
        pass
