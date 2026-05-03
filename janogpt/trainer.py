"""
Unified trainer for janogpt.

Supports:
- Single and multi-device training (CPU/GPU/TPU)
- Gradient accumulation (targeting 0.5M tokens per step)
- JIT compilation for single device, pmap for multi-device
- WandB logging
- Orbax checkpointing
- Evaluation outside of JIT
"""

from pathlib import Path
from typing import Dict, Iterator
import time
import numpy as np

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
import flax.linen as nn

from janogpt.config import Config
from janogpt.model import GPT, count_params
from janogpt.utils import shard_accum_batch


class Trainer:
    """
    Unified trainer for janogpt with model initialization and training.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        evaluators: list = None,
        logger = None,
        seed: int = 3407
    ):
        """
        Initialize trainer with dependency injection.

        Args:
            model: Model instance
            config: Training configuration
            evaluators: List of Evaluator instances
            logger: Logger instance
            seed: Random seed
        """
        self.config = config
        self.seed = seed
        self.model = model
        self.evaluators = evaluators or []
        self.logger = logger
        self.rng = jax.random.key(seed)

        # Device detection
        self.num_devices = jax.local_device_count()
        self.device_type = jax.devices()[0].platform
        print(f"Detected {self.num_devices} {self.device_type.upper()} device(s)")

        # Calculate effective batch size
        self.effective_batch_size = (
            config.micro_batch_size
            * config.gradient_accumulation_steps
            * self.num_devices
        )
        self.effective_batch_tokens = self.effective_batch_size * config.seq_len

        print(f"Effective batch: {self.effective_batch_size} seqs "
              f"= {self.effective_batch_tokens / 1e3:.0f}K tokens per step")

        # Initialize model and training state
        print("Initializing model...")
        self.state = self.create_train_state()

        # Initialize logger
        if self.logger:
            params = self.state.params
            param_count = count_params(params)
            self.logger.init(config, param_count)

        # Compile training steps
        self._compile_train_steps()

    def create_model(self) -> nn.Module:
        """Initialize GPT model."""
        return GPT(self.config)

    def create_learning_rate_schedule(self):
        """Create learning rate schedule (warmup + cosine decay or constant)."""
        if not self.config.use_lr_schedule:
            # Constant learning rate
            return self.config.learning_rate

        # Warmup + cosine decay schedule
        # Ensure warmup_steps < max_steps for valid decay schedule
        warmup_steps = min(self.config.warmup_steps, self.config.max_steps - 1)
        if warmup_steps < self.config.warmup_steps:
            print(f"⚠️  Reduced warmup_steps from {self.config.warmup_steps} to {warmup_steps} (max_steps={self.config.max_steps})")

        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=self.config.max_steps,
            end_value=self.config.min_learning_rate,
        )

    def create_optimizer(self):
        """Create AdamW optimizer with gradient clipping and learning rate schedule."""
        schedule = self.create_learning_rate_schedule()

        return optax.chain(
            optax.clip_by_global_norm(self.config.grad_clip),
            optax.adamw(
                learning_rate=schedule,
                b1=self.config.beta1,
                b2=self.config.beta2,
                eps=1e-8,
                weight_decay=self.config.weight_decay,
            ),
        )

    def create_train_state(self) -> train_state.TrainState:
        """Create TrainState with initialized parameters and optimizer."""
        # Initialize model
        self.rng, params_rng, dropout_rng = jax.random.split(self.rng, 3)

        # Dummy input for initialization
        dummy_input = jnp.zeros((1, self.config.seq_len), dtype=jnp.uint16)

        # Initialize parameters
        variables = self.model.init(
            {'params': params_rng, 'dropout': dropout_rng},
            dummy_input,
            inference=True,
        )
        params = variables['params']

        # Count parameters
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
        print(f"Model parameters: {param_count / 1e6:.2f}M")

        # Create optimizer
        tx = self.create_optimizer()

        # Create TrainState
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=tx,
        )

        # Replicate across devices if multi-device
        if self.num_devices > 1:
            state = self.replicate(state)
            print(f"State replicated across {self.num_devices} devices")

        return state

    def compute_loss(
        self,
        params,
        batch: Dict[str, jnp.ndarray],
        rng: jax.random.PRNGKey,
        training: bool = True,
    ):
        """
        Compute cross-entropy loss for next-token prediction.

        Args:
            params: Model parameters
            batch: dict with 'input_ids' (B, T)
            rng: Random key for dropout
            training: If True, use dropout

        Returns:
            loss: scalar
        """
        # Forward pass
        logits = self.model.apply(
            {'params': params},
            batch['input_ids'],
            inference=not training,
            rngs={'dropout': rng} if training else None,
        )

        # Shift for next-token prediction
        # logits[:, :-1, :] predicts tokens[:, 1:]
        shift_logits = logits[:, :-1, :]  # (B, T-1, V)
        shift_labels = batch['input_ids'][:, 1:]  # (B, T-1)

        # Cross-entropy loss
        loss = optax.softmax_cross_entropy_with_integer_labels(
            shift_logits, shift_labels
        ).mean()

        return loss

    def get_learning_rate(self, step):
        """Get current learning rate from optimizer schedule."""
        if not self.config.use_lr_schedule:
            # Constant learning rate
            return self.config.learning_rate

        # Evaluate schedule at step
        schedule = self.create_learning_rate_schedule()
        return schedule(step)

    # ========== Single Device Training (JIT) ==========

    def _train_step_single(
        self,
        state: train_state.TrainState,
        batches: Dict[str, jnp.ndarray],  # (accum_steps, micro_batch, seq)
        dropout_rngs: jax.random.PRNGKey,  # (accum_steps, 2)
    ):
        """
        Single-device train step with gradient accumulation.
        Uses lax.scan for memory efficiency.
        """
        def accum_fn(carry, xs):
            acc_grads, acc_loss = carry
            micro_batch, rng = xs

            # Compute loss and gradients for this micro-batch
            loss, grads = jax.value_and_grad(self.compute_loss)(
                state.params, micro_batch, rng, training=True
            )

            # Accumulate
            acc_grads = jax.tree_util.tree_map(
                lambda a, g: a + g, acc_grads, grads
            )
            acc_loss = acc_loss + loss

            return (acc_grads, acc_loss), None

        # Initialize accumulators
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        init_carry = (zero_grads, jnp.array(0.0))

        # Scan over accumulation steps
        (acc_grads, acc_loss), _ = jax.lax.scan(
            accum_fn,
            init_carry,
            (batches, dropout_rngs),
            length=self.config.gradient_accumulation_steps,
        )

        # Average gradients
        acc_grads = jax.tree_util.tree_map(
            lambda g: g / self.config.gradient_accumulation_steps,
            acc_grads
        )
        acc_loss = acc_loss / self.config.gradient_accumulation_steps

        # Apply gradients
        state = state.apply_gradients(grads=acc_grads)

        # Compute metrics
        grad_norm = optax.global_norm(acc_grads)
        metrics = {
            'loss': acc_loss,
            'perplexity': jnp.exp(acc_loss),
            'grad_norm': grad_norm,
            'learning_rate': self.get_learning_rate(state.step),
        }

        return state, metrics

    # ========== Multi-Device Training (pmap) ==========

    def _train_step_multi(
        self,
        state: train_state.TrainState,
        batches: Dict[str, jnp.ndarray],  # (accum, micro, seq) per device
        dropout_rngs: jax.random.PRNGKey,  # (accum, 2) per device
    ):
        """
        Multi-device train step with gradient accumulation.
        Each device accumulates independently, then pmean syncs.
        """
        def accum_fn(carry, xs):
            acc_grads, acc_loss = carry
            micro_batch, rng = xs

            loss, grads = jax.value_and_grad(self.compute_loss)(
                state.params, micro_batch, rng, training=True
            )

            acc_grads = jax.tree_util.tree_map(
                lambda a, g: a + g, acc_grads, grads
            )
            acc_loss = acc_loss + loss

            return (acc_grads, acc_loss), None

        # Initialize
        zero_grads = jax.tree_util.tree_map(jnp.zeros_like, state.params)
        init_carry = (zero_grads, jnp.array(0.0))

        # Accumulate on each device
        (acc_grads, acc_loss), _ = jax.lax.scan(
            accum_fn,
            init_carry,
            (batches, dropout_rngs),
            length=self.config.gradient_accumulation_steps,
        )

        # Average within device
        acc_grads = jax.tree_util.tree_map(
            lambda g: g / self.config.gradient_accumulation_steps,
            acc_grads
        )
        acc_loss = acc_loss / self.config.gradient_accumulation_steps

        # Sync across devices
        acc_grads = jax.lax.pmean(acc_grads, axis_name='devices')
        acc_loss = jax.lax.pmean(acc_loss, axis_name='devices')

        # Apply gradients (synchronized)
        state = state.apply_gradients(grads=acc_grads)

        grad_norm = optax.global_norm(acc_grads)
        metrics = {
            'loss': acc_loss,
            'perplexity': jnp.exp(acc_loss),
            'grad_norm': grad_norm,
            'learning_rate': self.get_learning_rate(state.step),
        }

        return state, metrics

    def _compile_train_steps(self):
        """Compile training steps based on number of devices."""
        if self.num_devices == 1:
            # Single device: use JIT
            self._train_step_fn = jax.jit(self._train_step_single)
            print("Using JIT compilation for single device")
        else:
            # Multi-device: use pmap
            self._train_step_fn = jax.pmap(
                self._train_step_multi,
                axis_name='devices',
                donate_argnums=(0,),  # Donate state to avoid copy
            )
            print(f"Using pmap compilation for {self.num_devices} devices")

    def _train_step(self, batch: Dict[str, jnp.ndarray]):
        """Unified train step that routes to compiled function."""
        if not hasattr(self, '_train_step_fn'):
            self._compile_train_step()

        # Prepare batch (includes sharding for multi-device)
        prepared_batch = self._prepare_batch(batch)

        # Execute compiled train step
        if self.num_devices == 1:
            # Single device
            metrics = self._train_step_fn(self.state, prepared_batch)
            self.state = metrics['state']
            return {k: v for k, v in metrics.items() if k != 'state'}
        else:
            # Multi-device
            metrics = self._train_step_fn(self.state, prepared_batch)
            self.state = metrics['state']
            # Unreplicate metrics
            return {k: self.unreplicate(v) if k != 'state' else v
                    for k, v in metrics.items() if k != 'state'}

    # ========== Evaluation ==========

    def _eval_step_single(
        self,
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
    ):
        """Evaluation step for single device."""
        loss = self.compute_loss(
            state.params,
            batch,
            None,  # No RNG needed for eval
            training=False,
        )
        return {'eval_loss': loss, 'eval_perplexity': jnp.exp(loss)}

    def _eval_step_multi(
        self,
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
    ):
        """Evaluation step for multiple devices."""
        loss = self.compute_loss(
            state.params,
            batch,
            None,
            training=False,
        )
        # Sync across devices
        loss = jax.lax.pmean(loss, axis_name='devices')
        return {'eval_loss': loss, 'eval_perplexity': jnp.exp(loss)}

    def _eval_step(self, batch: Dict[str, jnp.ndarray]):
        """Unified eval step that routes to single or multi device."""
        if self.num_devices == 1:
            metrics = self._eval_step_single(self.state, batch)
            return metrics['eval_loss']
        else:
            metrics = self._eval_step_multi(self.state, batch)
            return self.unreplicate(metrics['eval_loss'])

    def evaluate(self, eval_loader: Iterator, step = None, eval_iters = None) -> Dict[str, float]:
        """
        Run evaluation on validation set.

        Args:
            eval_loader: Iterator yielding validation batches
            step: Current training step (for logging)
            eval_iters: Number of batches to evaluate (overrides config)

        Returns:
            dict of evaluation metrics
        """
        if eval_iters is None:
            eval_iters = self.config.eval_iters
        # Compile eval step if not done yet
        if not hasattr(self, '_eval_step_fn'):
            if self.num_devices == 1:
                self._eval_step_fn = jax.jit(self._eval_step_single)
            else:
                self._eval_step_fn = jax.pmap(
                    self._eval_step_multi,
                    axis_name='devices'
                )

        total_loss = 0.0
        num_batches = 0

        for i, batch in enumerate(eval_loader):
            if i >= eval_iters:
                break

            # Convert to JAX arrays
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}

            # Shard if multi-device
            if self.num_devices > 1:
                from dataloader import shard_batch
                batch_jax = shard_batch(batch_jax, self.num_devices)

            # Eval step
            metrics = self._eval_step_fn(self.state, batch_jax)

            # Extract scalar (unreplicate if multi-device)
            if self.num_devices > 1:
                loss = float(metrics['eval_loss'][0])
            else:
                loss = float(metrics['eval_loss'])

            total_loss += loss
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_perplexity = jnp.exp(avg_loss)

        result = {
            'eval/loss': float(avg_loss),
            'eval/perplexity': float(avg_perplexity),
        }

        # Print
        step_str = f"step={step:>7d}" if step is not None else "final"
        print(
            f"[eval  {step_str}] "
            f"loss={avg_loss:.4f}  "
            f"ppl={avg_perplexity:.2f}"
        )

        return result

    # ========== Training Loop ==========

    def _prepare_batch(self, batch):
        """
        Prepare batch for training.
        Converts to JAX arrays and shards across devices if needed.
        """
        # Convert to JAX
        batch_jax = {k: jnp.array(v) for k, v in batch.items()}

        # Shard for gradient accumulation
        if self.num_devices > 1:
            batch_jax = shard_accum_batch(
                batch_jax,
                self.num_devices,
                self.config.gradient_accumulation_steps,
            )
        else:
            # Reshape for accumulation: (total, seq) -> (accum, micro, seq)
            def _reshape(x):
                B = x.shape[0]
                micro = B // self.config.gradient_accumulation_steps
                return x.reshape(
                    self.config.gradient_accumulation_steps,
                    micro,
                    *x.shape[1:]
                )
            batch_jax = {k: _reshape(v) for k, v in batch_jax.items()}

        return batch_jax

    def _generate_dropout_rngs(self):
        """Generate dropout RNGs for gradient accumulation steps."""
        self.rng, base = jax.random.split(self.rng)

        if self.num_devices > 1:
            # (num_devices, accum_steps) - each entry is a full PRNG key
            keys = jax.random.split(
                base,
                self.num_devices * self.config.gradient_accumulation_steps
            )
            return keys.reshape(
                self.num_devices,
                self.config.gradient_accumulation_steps,
            )
        else:
            # (accum_steps,) - each entry is a full PRNG key
            keys = jax.random.split(base, self.config.gradient_accumulation_steps)
            return keys

    def train(self, train_loader: Iterator):
        """
        Main training loop.

        Args:
            train_loader: Iterator yielding training batches
        """

        # Load checkpoint if resuming
        start_step = 0
        if self.config.resume_from_checkpoint:
            start_step = self.load_checkpoint()

        # Training loop
        t0 = time.perf_counter()
        loss_ema = None
        ema_alpha = 0.98

        print("\nStarting training...")
        print("=" * 80)

        for step, batch in enumerate(train_loader, start=start_step + 1):
            if step > self.config.max_steps:
                break

            # Prepare batch
            batch_jax = self._prepare_batch(batch)

            # Generate dropout RNGs
            dropout_rngs = self._generate_dropout_rngs()

            # Train step
            if step == 1:
                print(f"[step {step}] Starting first train step (JIT compilation will occur)...")
                step_start = time.perf_counter()

            self.state, metrics = self._train_step_fn(
                self.state,
                batch_jax,
                dropout_rngs
            )

            if step == 1:
                step_time = time.perf_counter() - step_start
                print(f"[step {step}] First step complete (JIT compile + execution: {step_time:.1f}s)")

            # Extract metrics (unreplicate if multi-device)
            if self.num_devices > 1:
                metrics = {k: float(v[0]) for k, v in metrics.items()}
            else:
                metrics = {k: float(v) for k, v in metrics.items()}

            # Update EMA loss
            loss = metrics['loss']
            loss_ema = loss if loss_ema is None else (
                ema_alpha * loss_ema + (1 - ema_alpha) * loss
            )

            # Log metrics
            if step % self.config.log_interval == 0:
                elapsed = time.perf_counter() - t0
                tokens_per_sec = step * self.effective_batch_tokens / elapsed

                log_dict = {
                    'train/loss': loss,
                    'train/loss_ema': loss_ema,
                    'train/perplexity': metrics['perplexity'],
                    'train/grad_norm': metrics['grad_norm'],
                    'train/learning_rate': metrics['learning_rate'],
                    'train/tokens_per_sec': tokens_per_sec,
                    'step': step,
                }

                if self.logger:
                    self.logger.log(log_dict, step=step)

            # Run evaluators
            if self.evaluators and step % self.config.eval_interval == 0:
                for evaluator in self.evaluators:
                    eval_metrics = evaluator.evaluate(
                        self.state,
                        self.compute_loss,
                        step
                    )
                    if self.logger:
                        self.logger.log(eval_metrics, step=step)
                    print(f"[eval  step={step:7d}] {evaluator.name}: " +
                          "  ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()]))

            # Save checkpoint
            if step % self.config.save_interval == 0:
                self.save_checkpoint(step)

        # Final evaluation and checkpoint
        print("=" * 80)
        print("Training complete! Running final evaluation...")
        if self.evaluators:
            for evaluator in self.evaluators:
                eval_metrics = evaluator.evaluate(
                    self.state,
                    self.compute_loss,
                    self.config.max_steps
                )
                if self.logger:
                    self.logger.log(eval_metrics, step=self.config.max_steps)
                print(f"[final eval] {evaluator.name}: " +
                      "  ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()]))

        # Save final checkpoint (only if not already saved)
        if self.config.max_steps % self.config.save_interval != 0:
            self.save_checkpoint(self.config.max_steps)

        # Finish logging
        if self.logger:
            self.logger.finish()

    # ========== Checkpointing ==========

    def save_checkpoint(self, step: int):
        """Save checkpoint at given step."""
        import orbax.checkpoint as ocp

        ckpt_dir = Path(self.config.output_dir).resolve() / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Unreplicate if multi-device
        state = self.unreplicate(self.state) if self.num_devices > 1 else self.state

        # Save with Orbax (convert config to dict for serialization)
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(
            str(ckpt_dir / f"step_{step}"),  # Orbax requires absolute path as string
            {
                'state': state,
                'step': step,
                'config': self.config.to_dict(),  # Convert Config to dict
                'rng': self.rng,
            }
        )
        print(f"✓ Checkpoint saved at step {step} to {ckpt_dir / f'step_{step}'}")

    def load_checkpoint(self, step = None) -> int:
        """
        Load checkpoint from step (or latest if None).

        Returns:
            step number of loaded checkpoint
        """
        import orbax.checkpoint as ocp

        ckpt_dir = Path(self.config.output_dir).resolve() / "checkpoints"

        if step is None:
            # Find latest
            checkpoints = sorted(ckpt_dir.glob("step_*"))
            if not checkpoints:
                raise ValueError(f"No checkpoints found in {ckpt_dir}")
            step = int(checkpoints[-1].name.split("_")[1])

        # Restore
        checkpointer = ocp.PyTreeCheckpointer()
        restored = checkpointer.restore(str(ckpt_dir / f"step_{step}"))

        self.state = restored['state']
        self.rng = restored['rng']

        # Replicate if multi-device
        if self.num_devices > 1:
            self.state = self.replicate(self.state)

        print(f"✓ Checkpoint loaded from step {step}")
        return step

    # ========== WandB ==========

    def init_wandb(self):
        """Initialize WandB run with config."""
        import wandb

        run_name = self.config.wandb_run_name
        if run_name is None:
            run_name = f"gpt2-{self.num_devices}x{self.device_type}"

        self.wandb_run = wandb.init(
            project=self.config.wandb_project,
            name=run_name,
            config={
                "model": "gpt2-124M",
                "vocab_size": self.config.voc_size,
                "num_blocks": self.config.num_blocks,
                "emb_dim": self.config.emb_dim,
                "num_heads": self.config.num_heads,
                "seq_len": self.config.seq_len,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "micro_batch_size": self.config.micro_batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "effective_batch_size": self.effective_batch_size,
                "effective_batch_tokens": self.effective_batch_tokens,
                "num_devices": self.num_devices,
                "device_type": self.device_type,
                "max_steps": self.config.max_steps,
            }
        )

        # Log code
        wandb.save(str(Path(__file__).parent / "model.py"))
        wandb.save(str(Path(__file__).parent / "config.py"))
        wandb.save(str(Path(__file__).parent / "trainer.py"))

        print(f"WandB initialized: {run_name}")

    # ========== Utilities ==========

    def replicate(self, tree):
        """Copy a pytree to all devices."""
        return jax.tree_util.tree_map(
            lambda x: jnp.array([x] * self.num_devices),
            tree
        )

    def unreplicate(self, tree):
        """Take device 0's copy (all devices hold same values after pmean)."""
        return jax.tree_util.tree_map(lambda x: x[0], tree)
