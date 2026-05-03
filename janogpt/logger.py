"""
Logging and evaluation implementations.
"""

from typing import Any, Callable, Dict

import jax.numpy as jnp

from janogpt.protocols import DataLoader
from janogpt.protocols import Evaluator as BaseEvaluator
from janogpt.protocols import Logger as BaseLogger

# ========== Evaluators ==========


class DatasetEvaluator(BaseEvaluator):
    """Evaluates loss on a validation dataset."""

    def __init__(
        self,
        data_loader: DataLoader,
        num_batches: int,
        name: str = "val",
    ):
        """
        Args:
            data_loader: DataLoader for validation data
            num_batches: Number of batches to evaluate
            name: Name for this evaluator (for logging)
        """
        self.data_loader = data_loader
        self.num_batches = num_batches
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(
        self,
        state,
        compute_loss_fn: Callable,
        step: int,
    ) -> Dict[str, float]:
        """Compute average loss over validation batches."""
        losses = []

        for _ in range(self.num_batches):
            batch = self.data_loader.get_batch()
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}

            # Eval mode (no dropout, no rng needed)
            loss = compute_loss_fn(state.params, batch_jax, None, training=False)
            losses.append(float(loss))

        avg_loss = sum(losses) / len(losses)
        perplexity = jnp.exp(avg_loss)

        return {
            f"{self.name}/loss": avg_loss,
            f"{self.name}/perplexity": float(perplexity),
        }


# ========== Loggers ==========


class ConsoleLogger(BaseLogger):
    """Simple console logger."""

    def __init__(self):
        self.enabled = True

    def init(self, config: Any, model_params: int):
        """Print initialization info."""
        print(f"✓ Console logger initialized (model: {model_params / 1e6:.2f}M params)")

    def log(self, metrics: Dict[str, Any], step: int):
        """Print metrics to console."""
        if not self.enabled:
            return

        # Format metrics nicely
        metric_str = "  ".join(
            [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        )
        print(f"[step {step:7d}] {metric_str}")

    def finish(self):
        """Nothing to clean up."""
        pass


class WandBLogger(BaseLogger):
    """Weights & Biases logger."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.run = None

    def init(self, config: Any, model_params: int):
        """Initialize wandb run."""
        if not self.enabled:
            print("✓ WandB logging disabled")
            return

        import wandb

        # Build run config
        run_config = {
            "model_params": model_params,
            "learning_rate": config.learning_rate,
            "batch_size": config.micro_batch_size * config.gradient_accumulation_steps,
            "max_steps": config.max_steps,
        }

        # Add all config fields
        if hasattr(config, "to_dict"):
            run_config.update(config.to_dict())

        self.run = wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            entity=getattr(config, "wandb_entity", None),
            tags=getattr(config, "wandb_tags", []),
            config=run_config,
        )
        print(f"✓ WandB initialized: {self.run.url}")

    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics to wandb."""
        if not self.enabled or not self.run:
            return

        import wandb

        wandb.log(metrics, step=step)

    def finish(self):
        """Finish wandb run."""
        if self.run:
            import wandb

            wandb.finish()
            print("✓ WandB run finished")


class MultiLogger(BaseLogger):
    """Combines multiple loggers."""

    def __init__(self, loggers: list):
        self.loggers = loggers

    def init(self, config: Any, model_params: int):
        for logger in self.loggers:
            logger.init(config, model_params)

    def log(self, metrics: Dict[str, Any], step: int):
        for logger in self.loggers:
            logger.log(metrics, step)

    def finish(self):
        for logger in self.loggers:
            logger.finish()
