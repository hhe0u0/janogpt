from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import jax.numpy as jnp


@dataclass
class Config:
    # Model architecture
    dropout_prob: float = 0.1
    num_blocks: int = 12
    emb_dim: int = 768
    ff_dim: int = 768 * 4 # Computed from emb_dim if None
    num_heads: int = 12
    seq_len: int = 1024
    epsilon: float = 1e-6
    vocab_size: int = 50304

    # Optimizer
    learning_rate: float = 6e-4
    min_learning_rate: float = 6e-5
    warmup_steps: int = 2000
    use_lr_schedule: bool = True  # If False, use constant learning_rate
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    weight_decay: float = 0.1

    # Training
    max_steps: int = 600000
    micro_batch_size: int = 4  # Per device
    gradient_accumulation_steps: int = 16
    seed: int = 42

    # Evaluation & Logging
    eval_interval: int = 1000
    eval_iters: int = 200
    log_interval: int = 10

    # Checkpointing
    save_interval: int = 5000
    output_dir: str = "output"
    resume_from_checkpoint: Optional[str] = None

    # Data
    data_dir: str = "data"
    dataloader_class: str = "FileDataLoader"  # "FileDataLoader" or "DummyDataLoader"

    # WandB
    wandb_project: str = "janogpt"
    wandb_run_name: Optional[str] = None
    wandb_log: bool = True
    wandb_entity: Optional[str] = None
    wandb_tags: list = field(default_factory=list)

    # Dtype
    dtype = jnp.bfloat16
    param_dtype = jnp.float32

    def __post_init__(self):
        """Post-initialization to compute derived fields."""
        if self.ff_dim is None:
            self.ff_dim = 4 * self.emb_dim

    @classmethod
    def from_json(cls, json_path: str) -> "Config":
        """
        Load configuration from JSON file.

        Args:
            json_path: Path to JSON config file

        Returns:
            Config instance
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")

        with open(json_path, 'r') as f:
            data = json.load(f)

        # Flatten nested structure
        flat_config = {}

        # Model parameters
        if "model" in data:
            flat_config.update(data["model"])

        # Optimizer parameters
        if "optimizer" in data:
            flat_config.update(data["optimizer"])

        # Training parameters
        if "training" in data:
            flat_config.update(data["training"])

        # Data parameters
        if "data" in data:
            flat_config.update(data["data"])  # Include all data fields

        # Logging parameters
        if "logging" in data:
            flat_config.update(data["logging"])

        # Checkpointing parameters
        if "checkpointing" in data:
            flat_config.update(data["checkpointing"])

        # WandB parameters
        if "wandb" in data:
            wandb_config = data["wandb"]
            flat_config["wandb_log"] = wandb_config.get("enabled", True)
            flat_config["wandb_project"] = wandb_config.get("project", "janogpt")
            flat_config["wandb_run_name"] = wandb_config.get("run_name", None)
            flat_config["wandb_entity"] = wandb_config.get("entity", None)
            flat_config["wandb_tags"] = wandb_config.get("tags", [])

        # Remove any keys that aren't Config fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        flat_config = {k: v for k, v in flat_config.items() if k in valid_fields}

        return cls(**flat_config)

    def to_dict(self) -> dict:
        """Convert config to dictionary (excluding non-serializable fields)."""
        return {
            "model": {
                "dropout_prob": self.dropout_prob,
                "num_blocks": self.num_blocks,
                "emb_dim": self.emb_dim,
                "ff_dim": self.ff_dim,
                "num_heads": self.num_heads,
                "seq_len": self.seq_len,
                "epsilon": self.epsilon,
                "vocab_size": self.vocab_size,
            },
            "optimizer": {
                "learning_rate": self.learning_rate,
                "min_learning_rate": self.min_learning_rate,
                "warmup_steps": self.warmup_steps,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "grad_clip": self.grad_clip,
                "weight_decay": self.weight_decay,
            },
            "training": {
                "max_steps": self.max_steps,
                "micro_batch_size": self.micro_batch_size,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "seed": self.seed,
            },
            "data": {
                "data_dir": self.data_dir,
            },
            "logging": {
                "eval_interval": self.eval_interval,
                "eval_iters": self.eval_iters,
                "log_interval": self.log_interval,
            },
            "checkpointing": {
                "save_interval": self.save_interval,
                "output_dir": self.output_dir,
                "resume_from_checkpoint": self.resume_from_checkpoint,
            },
            "wandb": {
                "enabled": self.wandb_log,
                "project": self.wandb_project,
                "run_name": self.wandb_run_name,
                "entity": self.wandb_entity,
                "tags": self.wandb_tags,
            },
        }

    def save_json(self, json_path: str):
        """Save configuration to JSON file."""
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)

        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __repr__(self):
        """Pretty print configuration."""
        lines = ["Config("]
        lines.append("  Model:")
        lines.append(f"    blocks={self.num_blocks}, emb_dim={self.emb_dim}, heads={self.num_heads}")
        lines.append(f"    seq_len={self.seq_len}, vocab={self.vocab_size}")
        lines.append("  Training:")
        lines.append(f"    max_steps={self.max_steps}, lr={self.learning_rate}")
        lines.append(f"    micro_batch={self.micro_batch_size}, accum={self.gradient_accumulation_steps}")
        lines.append("  Data:")
        lines.append(f"    data_dir={self.data_dir}")
        lines.append("  Output:")
        lines.append(f"    output_dir={self.output_dir}")
        lines.append(")")
        return "\n".join(lines)