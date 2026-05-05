# Vibe Coding with JAX: How I Turned 3 Hours of Tests Into 12 Hours of Debugging

*Or: 11 footguns I found while training GPT-2 on TPUs*

---

You know that feeling when you're *in the zone*? Code is flowing, your model is training, the loss is going down. You're vibing. You're shipping. Who needs tests when the vibes are this good?

Yeah, about that.

I spent two days and 78 git commits debugging issues that would have taken 3 hours to prevent with proper tests. Here's what I learned about JAX's sharp edges, so you don't have to learn them the hard way.

## The Setup: Just Vibing with JAX

I was building a GPT-2 trainer in JAX. Standard stuff:
- Multi-device training (because TPUs are cool)
- Gradient accumulation (because big batches are efficient)  
- Checkpointing (because training is expensive)
- Mixed precision (because we're not made of memory)

JAX is amazing for ML — functional, composable, hardware-agnostic. But it's also sharp. Very sharp. And when you're vibe coding without tests, those sharp edges will find you.

Here are the 11 ways JAX bit me.

---

## Pitfall #1: The Shape Shifter

**The Vibe**: "Multi-device training is just `pmap` right? Ship it!"

**The Reality**:
```python
# Training (multi-device)
state = trainer.train()  # state.params.shape = (2, 50304, 768)

# Evaluation (single-device)  
evaluator.evaluate(state, ...)  
# 💥 ScopeParamShapeError: expected (50304, 768), got (2, 50304, 768)
```

**What Happened**: JAX's `pmap` replicates parameters across devices. Your params aren't `(vocab, dim)` anymore — they're `(num_devices, vocab, dim)`. When you pass this to single-device evaluation, JAX gets confused.

**The Fix**:
```python
# Unreplicate before single-device operations
eval_state = jax.tree_util.tree_map(lambda x: x[0], state) if multi_device else state
evaluator.evaluate(eval_state, ...)
```

**The Lesson**: Multi-device state is a different type. Treat it as such. Use type hints:
```python
def evaluate(self, state: SingleDeviceState, ...) -> dict[str, float]:
    """Note: Expects UNREPLICATED state."""
```

**Prevention**: An integration test would have caught this in 30 seconds.

---

## Pitfall #2: The Memory Magician

**The Vibe**: "Training works on GPU, should work on TPU!"

**The Reality**:
```
RESOURCE_EXHAUSTED: Error allocating device buffer: 
Attempting to allocate 12.00G. That was not possible.
There are 10.40G free.
```

**What Happened**: I had this beautiful mental model of memory usage:

```python
# My calculation
memory = 40 × params  # Where did 40 come from? 🤷

# Reality (AdamW in float32)
memory = 4 × params   # params + grads + momentum + variance
```

Off by 10x! But wait, there's more. Training and evaluation have *different* memory profiles:

```python
# Training with gradient accumulation
for micro_batch in batches:
    process(micro_batch)  # One micro-batch at a time: ~0.4GB
    accumulate_grads()

# Evaluation (naive)
process(entire_batch)  # All 512 sequences at once: 12GB 💥
```

**The Fix**:
```python
# Separate batch sizes!
train_loader = DataLoader(batch_size=512)  # Processed incrementally
eval_loader = DataLoader(batch_size=8)     # Processed all at once
```

**The Lesson**: Memory profiling isn't optional. Your mental model is wrong until proven otherwise.

**Prevention**: A smoke test on the target hardware would have revealed this immediately.

---

## Pitfall #3: The String Sorter

**The Vibe**: "Just sort the checkpoint folders and take the last one!"

**The Reality**:
```python
checkpoints = sorted(Path("output/checkpoints").glob("step_*"))
latest = checkpoints[-1]

# What I expected: step_100
# What I got: step_50

# Why? String sorting!
# "step_100" < "step_50"  (because "1" < "5")
```

**The Fix**:
```python
checkpoints = [(int(p.name.split("_")[1]), p) for p in Path(...).glob("step_*")]
latest = max(checkpoints)[0]
```

**The Lesson**: This is a Python 101 mistake. Don't trust string sorting for numbers.

**Prevention**: A test with checkpoints `step_10`, `step_20`, `step_100` would have caught this instantly.

---

## Pitfall #4: The Cleanup Catastrophe

**The Vibe**: "Let's be good citizens and clean up resources!"

**The Reality**:
```python
def cleanup(self):
    """Clean up after training."""
    del self.state
    del self._train_step_fn  # Compiled function
    gc.collect()
    
# First training run
trainer.train()  # ✓ Works
trainer.cleanup()  # ✓ Cleanup

# Resume from checkpoint
trainer.train()  # 💥 AttributeError: no attribute '_train_step_fn'
```

**What Happened**: I manually deleted the compiled function. Then tried to use it again. Turns out, manual resource management is hard.

**The Fix**:
```python
def cleanup(self):
    """Clean up after training."""
    jax.clear_caches()  # Clear JAX's compilation cache
    gc.collect()        # Let Python GC handle the rest
    # Don't manually delete attributes!
```

**The Lesson**: Trust the garbage collector. Manual deletion should be rare and well-justified. If you *must* delete, check `hasattr()` before using.

**Prevention**: Test checkpoint resumption in the same process.

---

## Pitfall #5: The Serialization Shapeshifter

**The Vibe**: "Orbax handles checkpointing, just save and load!"

**The Reality**:
```python
# Save
checkpointer.save("step_100", {"state": train_state, ...})  # ✓

# Load  
restored = checkpointer.restore("step_100")
state = restored["state"]

# Try to use it
state.params  # 💥 AttributeError: 'dict' has no attribute 'params'
```

**What Happened**: Orbax saved my `TrainState` object but loaded it back as a plain dict. The type information was lost.

**The Fix**:
```python
# Manually reconstruct the object
state_dict = restored["state"]
state = train_state.TrainState(
    step=state_dict["step"],
    params=state_dict["params"],
    apply_fn=model.apply,
    tx=optimizer,
    opt_state=state_dict["opt_state"],  # Or create fresh — see next pitfall
)
```

**The Lesson**: Serialization loses type information. Always test save/load roundtrips.

**Prevention**: A simple test would have caught this:
```python
def test_checkpoint():
    save_checkpoint(state, "test")
    loaded_state = load_checkpoint("test")
    assert isinstance(loaded_state, train_state.TrainState)
    assert loaded_state.params == state.params
```

---

## Pitfall #6: The Optimizer Origami

**The Vibe**: "I'll just restore the optimizer state from the checkpoint!"

**The Reality**:
```python
# Restore optimizer state
state = train_state.TrainState(
    params=loaded["params"],
    opt_state=loaded["opt_state"],  # Nested dicts from Orbax
    tx=optimizer,
)

# Continue training
state = state.apply_gradients(grads)
# 💥 AttributeError: 'dict' has no attribute 'mu'
```

**What Happened**: Optax optimizers (like Adam) use NamedTuples with specific structure:
```python
AdamState(mu=..., nu=..., count=...)
```

But Orbax restores them as nested dicts:
```python
{"mu": ..., "nu": ..., "count": ...}
```

Dicts don't have `.mu` attributes. Optax expects NamedTuples.

**The Fix**: Just recreate the optimizer state fresh:
```python
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=loaded["params"],
    tx=optimizer,  # Fresh optimizer with fresh state
)
```

**The Tradeoff**: You lose momentum and variance from Adam. But:
1. The model adapts within a few steps
2. It's simpler and more robust
3. No fighting with NamedTuple reconstruction

**The Lesson**: Sometimes the simple solution beats the "correct" solution. Don't fight the framework.

**Prevention**: An end-to-end test: train → save → load → train more.

---

## Pitfall #7: The Import Illusion

**The Vibe**: "I imported `jax.numpy`, that's all I need!"

**The Reality**:
```python
import jax.numpy as jnp

# Later in the code...
probs = jax.nn.softmax(logits)  # 💥 NameError: 'jax' is not defined
```

**What Happened**: `jax.numpy` doesn't import the `jax` namespace. They're separate modules.

**The Fix**:
```python
import jax
import jax.numpy as jnp
```

**The Lesson**: This is embarrassing, but it happened in multiple files. Import discipline matters.

**Prevention**: Static analysis! Run `ruff` or `flake8`:
```bash
ruff check .  # Catches undefined names instantly
```

---

## Pitfall #8: The Log Logger

**The Vibe**: "More logging is better logging!"

**The Reality**:
```
[step 20] val/loss=8.6102 val/perplexity=5487.3760 ...
[eval step=20] val: val/loss=8.6102 val/perplexity=5487.3760 ...
```

**What Happened**: I had a logger that printed metrics. I also manually printed metrics. Now I have two logs.

**The Fix**: Single responsibility. The logger logs. The trainer trains.
```python
# Bad
metrics = evaluator.evaluate(state, ...)
logger.log(metrics)
print(f"[eval] {metrics}")  # Duplicate!

# Good
metrics = evaluator.evaluate(state, ...)
logger.log(metrics)  # Logger handles all printing
```

**The Lesson**: Don't Repeat Yourself. Especially in logging.

---

## Pitfall #9: The Cache Cascade

**The Vibe**: "JAX will manage its own memory!"

**The Reality**:
```
# After training completes...
# Running final evaluation...

RESOURCE_EXHAUSTED: Out of memory
```

**What Happened**: JAX's compilation cache accumulated during training. By the end, the cache + training state + evaluation batch exceeded available memory.

**The Fix**: Clear caches before memory-intensive operations:
```python
# Before final evaluation
jax.clear_caches()
evaluate(state)
```

**The Lesson**: JAX caches are memory. Budget for them.

**Prevention**: Memory monitoring during training:
```python
def log_memory(step):
    if step % 100 == 0:
        # Log cache size, heap size, GPU memory
        ...
```

---

## Pitfall #10: The Finally Fallacy

**The Vibe**: "`finally` blocks always run, perfect for cleanup!"

**The Reality**:
```python
def train(self):
    try:
        # Initialize
        self.state = self.create_train_state()  # Line 93
        
        # Train
        for step in range(max_steps):
            ...
    finally:
        # Run final evaluation
        eval_state = self.unreplicate(self.state)  # 💥 No attribute 'state'
```

**What Happened**: If initialization fails before line 93, `self.state` never gets created. The `finally` block runs anyway and tries to access it.

**The Fix**: Defensive programming in cleanup code:
```python
finally:
    if hasattr(self, 'state'):
        # Safe to use self.state
        ...
```

**The Lesson**: `finally` blocks run even when things go wrong. Guard all attribute access.

---

## Pitfall #11: The Target Token Tango

**The Vibe**: "Let me just quickly add dynamic batch calculation..."

**The Reality**:
```python
# Copy-pasted from another notebook
target_tokens = 512 * 1024  # 524,288 tokens

# But wait, I wanted 500K!
target_tokens = 500_000  # Much cleaner
```

**What Happened**: Magic numbers proliferate. Each notebook had slightly different target token counts: 524K, 500K, 512K. No single source of truth.

**The Fix**: Constants and documentation:
```python
# Why 500K? Trade-off between:
# - Smaller batches: More frequent updates, better generalization
# - Larger batches: Better GPU utilization, faster training
# We chose 500K as a sweet spot for our setup
TARGET_TOKENS_PER_STEP = 500_000
```

**The Lesson**: Document your magic numbers. Future you will thank past you.

---

## The Damage Report

Let's tally up the cost:

- **Total commits**: 78
- **Total time**: 16-20 hours of debugging
- **Preventable with tests**: 12-15 hours (60-75%)
- **Avoidable with better design**: 6 hours (30%)
- **Legitimate exploration**: 2 hours (10%)

**The punchline**: I spent 12 hours debugging issues that 3 hours of writing tests would have prevented.

---

## The Vibe Check: What I Learned

### 1. **Test Multi-Device Early**
JAX's multi-device support is magical, but the magic has sharp edges. Test that training and evaluation work together with replicated state.

```python
def test_multi_device_training_and_eval():
    # This test would have saved 3 hours
    trainer = Trainer(num_devices=2)
    trainer.train()
    trainer.evaluate()  # Would have failed immediately
```

### 2. **Checkpoint Save/Load is Not Optional Testing**
Checkpointing is complex. Serialization loses types. Optimizer state has structure. Test the full loop:

```python
def test_checkpoint_resume():
    # Train, save, load, train more
    trainer1 = Trainer()
    trainer1.train(steps=10)
    trainer1.save_checkpoint()
    
    trainer2 = Trainer(resume_from="checkpoint")
    trainer2.train(steps=10)  # Would have caught 4 bugs
```

### 3. **Memory Profiling is Not Optional**
Your mental model of memory usage is wrong. Profile early:

```python
# Before full training
profiler = MemoryProfiler()
profiler.estimate_memory(config)
profiler.verify_fits(device_memory_gb=32)

# Would have caught the eval OOM immediately
```

### 4. **Static Analysis is Free**
Set up `ruff` and `mypy`. They catch stupid mistakes instantly:

```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
```

### 5. **Smoke Tests on Target Hardware**
Run a quick 3-step training on the actual hardware before the full run:

```python
def smoke_test():
    config = Config(max_steps=3, eval_interval=2)
    trainer = Trainer(config)
    trainer.train()  # 2 minutes, catches 90% of issues
```

### 6. **Documentation is Code**
When you write a memory formula or batch size calculation, add a comment explaining *why*:

```python
# Memory for AdamW optimizer (float32):
# - Parameters: 1x (model weights)
# - Gradients: 1x (for backprop)  
# - Adam momentum: 1x (m state)
# - Adam variance: 1x (v state)
# Total: 4x parameters = ~2GB for GPT-2 124M
base_memory = 4 * param_count * 4  # 4 bytes per float32
```

### 7. **Trust the GC, Not Yourself**
Manual resource management is hard. Let Python's garbage collector do its job. Only manually delete if you can justify it:

```python
# Bad
del self.state
del self._train_step_fn

# Good
jax.clear_caches()  # Clear what JAX owns
gc.collect()        # Let Python handle the rest
```

---

## The Vibes Are Better With Tests

Look, I get it. Tests aren't sexy. They don't feel like progress. When you're in flow, stopping to write tests feels like killing the vibe.

But you know what *really* kills the vibe? Spending 12 hours debugging issues that 3 hours of tests would have prevented.

The fastest way to ship is to not break things. And the fastest way to not break things is to test them.

So next time you're vibing with JAX:

1. **Write the integration test first** — Multi-device? Test it. Checkpointing? Test it. Memory-intensive? Profile it.

2. **Set up static analysis** — `ruff` and `mypy` catch bugs before you even run the code.

3. **Smoke test on target hardware** — 3 minutes of testing beats 3 hours of debugging.

4. **Document the sharp edges** — Future you will thank past you.

5. **Ship tests, not just features** — Tests are the features that let you ship future features.

JAX is amazing. Multi-device training is magical. TPUs are fast. But none of that matters if you spend your time debugging instead of training.

Test your vibes. Ship confidently. And may your gradients always flow in the right direction.

---

## Resources

**The Code**: All the tests and fixes are available at [github.com/yourusername/janogpt](https://github.com)

**The Analysis**: Full debug session breakdown in `DEBUG_SESSION_ANALYSIS.md`

**The Tests**: 
- `test_checkpoint.py` - Basic checkpoint save/load
- `test_training_e2e.py` - Full training, eval, save, resume workflow

**JAX Resources**:
- [JAX Documentation](https://jax.readthedocs.io/)
- [Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html)
- [Multi-Device Guide](https://jax.readthedocs.io/en/latest/jax-101/06-parallelism.html)

---

*Have you hit JAX's sharp edges? Share your war stories in the comments. Let's help each other vibe more efficiently.*

---

**Update**: Thanks for all the comments! A few folks asked about the memory profiling setup — I'll write a follow-up post about building a proper memory profiler for JAX training. Subscribe to get notified!
