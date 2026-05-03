# Adding Custom Metrics to WandB

This guide shows how to add custom metrics to your training runs.

## Current Metrics

### Training Metrics (every `log_interval` steps)
- `train/loss` - Cross-entropy loss on current batch
- `train/loss_ema` - Exponential moving average of loss (smoothed)
- `train/perplexity` - exp(loss), interpretability metric
- `train/grad_norm` - Gradient norm before clipping
- `train/learning_rate` - Current learning rate from schedule
- `train/tokens_per_sec` - Training throughput
- `step` - Global training step

### Evaluation Metrics (every `eval_interval` steps)
- `val/loss` - Average loss on validation set
- `val/perplexity` - Validation perplexity

## How to Add Metrics

### Method 1: Training Step Metrics

Edit `janogpt/trainer.py` in the `_train_step_single()` and `_train_step_multi()` functions.

**Example: Add parameter norm**

```python
# In janogpt/trainer.py, around line 278-285
# Compute metrics
grad_norm = optax.global_norm(acc_grads)
param_norm = optax.global_norm(state.params)  # ← ADD THIS

metrics = {
    "loss": acc_loss,
    "perplexity": jnp.exp(acc_loss),
    "grad_norm": grad_norm,
    "param_norm": param_norm,  # ← ADD THIS
    "learning_rate": self.get_learning_rate(state.step),
}
```

**Do the same for `_train_step_multi()` around line 335.**

Then add to the logging dict around line 592:

```python
log_dict = {
    "train/loss": loss,
    "train/loss_ema": loss_ema,
    "train/perplexity": metrics["perplexity"],
    "train/grad_norm": metrics["grad_norm"],
    "train/param_norm": metrics["param_norm"],  # ← ADD THIS
    "train/learning_rate": metrics["learning_rate"],
    "train/tokens_per_sec": tokens_per_sec,
    "step": step,
}
```

### Method 2: System/Hardware Metrics

Add system metrics directly in the training loop:

```python
# In janogpt/trainer.py, around line 592
log_dict = {
    "train/loss": loss,
    # ... existing metrics ...
    
    # System metrics
    "system/step_time_ms": (elapsed / step) * 1000,
    "system/effective_batch_size": self.effective_batch_size,
    "system/tokens_per_step": self.effective_batch_tokens,
    "step": step,
}
```

### Method 3: Custom Evaluator

Create a new evaluator class in `janogpt/logger.py`:

```python
class DetailedEvaluator(BaseEvaluator):
    """Evaluates additional metrics like top-k accuracy."""
    
    def __init__(self, data_loader, model, num_batches, name="detailed"):
        self.data_loader = data_loader
        self.model = model
        self.num_batches = num_batches
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    def evaluate(self, state, compute_loss_fn, step):
        """Compute detailed metrics."""
        losses = []
        entropies = []
        
        for _ in range(self.num_batches):
            batch = self.data_loader.get_batch()
            batch_jax = {k: jnp.array(v) for k, v in batch.items()}
            
            # Get logits
            logits = self.model.apply(
                {'params': state.params},
                batch_jax['input_ids'],
                inference=True
            )
            
            # Compute loss
            loss = compute_loss_fn(state.params, batch_jax, None, training=False)
            losses.append(float(loss))
            
            # Compute entropy (measure of uncertainty)
            probs = jax.nn.softmax(logits, axis=-1)
            entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1).mean()
            entropies.append(float(entropy))
        
        return {
            f"{self.name}/loss": sum(losses) / len(losses),
            f"{self.name}/entropy": sum(entropies) / len(entropies),
        }
```

Then use it in your training script:

```python
from janogpt.logger import DetailedEvaluator

detailed_eval = DetailedEvaluator(
    data_loader=val_loader,
    model=model,
    num_batches=10,
    name="detailed_val"
)

trainer = Trainer(
    model=model,
    config=config,
    evaluators=[dataset_evaluator, detailed_eval],  # Add both
    logger=logger,
    seed=config.seed
)
```

## Useful Metrics to Add

### Gradient Analysis
```python
# Gradient statistics
grad_norm = optax.global_norm(acc_grads)
grad_max = jax.tree_util.tree_reduce(
    jnp.maximum,
    jax.tree_util.tree_map(lambda x: jnp.max(jnp.abs(x)), acc_grads)
)
grad_mean = jax.tree_util.tree_reduce(
    jnp.add,
    jax.tree_util.tree_map(lambda x: jnp.abs(x).mean(), acc_grads)
) / jax.tree_util.tree_reduce(lambda x, y: x + 1, acc_grads, 0)

metrics = {
    # ... existing ...
    "grad_norm": grad_norm,
    "grad_max": grad_max,
    "grad_mean": grad_mean,
}
```

### Parameter Statistics
```python
# Parameter statistics
param_norm = optax.global_norm(state.params)
param_max = jax.tree_util.tree_reduce(
    jnp.maximum,
    jax.tree_util.tree_map(lambda x: jnp.max(jnp.abs(x)), state.params)
)
param_mean = jax.tree_util.tree_reduce(
    jnp.add,
    jax.tree_util.tree_map(lambda x: jnp.abs(x).mean(), state.params)
) / jax.tree_util.tree_reduce(lambda x, y: x + 1, state.params, 0)

metrics = {
    # ... existing ...
    "param_norm": param_norm,
    "param_max": param_max,
    "param_mean": param_mean,
}
```

### Learning Dynamics
```python
# Gradient-to-parameter ratio (useful for learning rate tuning)
grad_to_param_ratio = grad_norm / (param_norm + 1e-8)

# Effective learning rate (LR × grad_norm)
effective_lr = metrics["learning_rate"] * grad_norm

metrics = {
    # ... existing ...
    "grad_to_param_ratio": grad_to_param_ratio,
    "effective_lr": effective_lr,
}
```

### Model Output Statistics
```python
# In compute_loss or a separate function
logits = model.apply({'params': params}, input_ids, inference=True)

# Prediction confidence (how confident is the model?)
probs = jax.nn.softmax(logits, axis=-1)
max_probs = jnp.max(probs, axis=-1)
avg_confidence = max_probs.mean()

# Prediction entropy (measure of uncertainty)
entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1).mean()

# Top-1 accuracy (% of correct next-token predictions)
predictions = jnp.argmax(logits[:, :-1, :], axis=-1)
targets = input_ids[:, 1:]
accuracy = (predictions == targets).mean()

return {
    "loss": loss,
    "confidence": avg_confidence,
    "entropy": entropy,
    "accuracy": accuracy,
}
```

## Performance Metrics

```python
import time

# Track step time
step_start = time.perf_counter()
state, metrics = train_step(...)
step_time = time.perf_counter() - step_start

log_dict = {
    # ... existing metrics ...
    "system/step_time_ms": step_time * 1000,
    "system/steps_per_sec": 1.0 / step_time,
    "system/tokens_per_sec": self.effective_batch_tokens / step_time,
}
```

## GPU Memory Tracking

```python
# Option 1: Using JAX device memory
import jax

device = jax.devices()[0]
mem_info = device.memory_stats()  # Returns dict with 'bytes_in_use', 'peak_bytes_in_use'

log_dict["system/gpu_memory_gb"] = mem_info['bytes_in_use'] / 1e9
log_dict["system/gpu_memory_peak_gb"] = mem_info['peak_bytes_in_use'] / 1e9

# Option 2: Using nvidia-smi (external command)
import subprocess

result = subprocess.run(
    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'],
    capture_output=True, text=True
)
gpu_mem_mb = int(result.stdout.strip().split('\n')[0])
log_dict["system/gpu_memory_mb"] = gpu_mem_mb
```

## Best Practices

1. **Don't log too frequently** - Default `log_interval=10` is good. Logging every step slows training.

2. **Group metrics by category** - Use prefixes:
   - `train/*` - Training metrics
   - `val/*` - Validation metrics  
   - `system/*` - System/hardware metrics
   - `grad/*` - Gradient statistics
   - `param/*` - Parameter statistics

3. **Avoid expensive metrics in training loop** - Compute heavy metrics only during evaluation.

4. **Use EMA for noisy metrics** - Smooth noisy signals like loss:
   ```python
   loss_ema = 0.98 * loss_ema + 0.02 * loss  # 98% previous, 2% current
   ```

5. **Log hardware info once at init** - Log static info (model size, batch size, devices) in WandB config, not every step.

## Example: Full Custom Metrics

```python
# In _train_step_single(), around line 278
# Compute comprehensive metrics
grad_norm = optax.global_norm(acc_grads)
param_norm = optax.global_norm(state.params)
grad_to_param = grad_norm / (param_norm + 1e-8)

metrics = {
    "loss": acc_loss,
    "perplexity": jnp.exp(acc_loss),
    "grad_norm": grad_norm,
    "param_norm": param_norm,
    "grad_to_param_ratio": grad_to_param,
    "learning_rate": self.get_learning_rate(state.step),
}

# In training loop, around line 592
log_dict = {
    "train/loss": loss,
    "train/loss_ema": loss_ema,
    "train/perplexity": metrics["perplexity"],
    "train/grad_norm": metrics["grad_norm"],
    "train/param_norm": metrics["param_norm"],
    "train/grad_to_param_ratio": metrics["grad_to_param_ratio"],
    "train/learning_rate": metrics["learning_rate"],
    "train/tokens_per_sec": tokens_per_sec,
    "system/step_time_ms": (elapsed / step) * 1000,
    "step": step,
}
```

This will give you a comprehensive view of your training dynamics in WandB!
