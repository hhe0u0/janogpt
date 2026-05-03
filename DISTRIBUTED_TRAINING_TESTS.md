# Distributed Training Tests

## Overview

Comprehensive integration tests for multi-device training using **simulated CPU devices** via XLA_FLAGS. No GPUs or TPUs required for CI.

## Test Suite

### 1. `test_multi_cpu_device_setup`

**Purpose**: Verify XLA_FLAGS creates multiple CPU devices

**How it works**:
```python
env["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
# JAX now sees 4 CPU devices instead of 1
```

**Validates**: Environment setup for distributed testing

---

### 2. `test_distributed_overfitting_single_batch` ⭐

**Purpose**: Gold standard test for distributed training correctness

**Strategy**: Overfit a single small batch repeatedly
- If loss drops quickly → backprop, pmap, replication all work correctly
- If loss stays high → something is broken in distributed setup

**Configuration**:
- **Devices**: 4 CPU cores (simulated)
- **Model**: Tiny GPT (2 blocks, 64 dim, 0.84M params)
- **Training**: 50 steps, learning_rate=3e-3
- **Batch**: Single batch (4 sequences, 32 tokens each)
- **Dropout**: 0.0 (deterministic overfitting)

**Expected Results**:
```
Initial loss: 5.29
Final loss:   0.018
Reduction:    99.7%
```

**Why this works**: 
- Small model can memorize small batch perfectly
- High LR ensures fast convergence
- 99%+ loss reduction proves distributed training works end-to-end

---

### 3. `test_gradient_accumulation_multi_device`

**Purpose**: Test gradient accumulation with pmap

**Configuration**:
- **Devices**: 2 CPUs
- **gradient_accumulation_steps**: 4
- **Effective batch**: micro_batch × accum × devices = 1 × 4 × 2 = 8

**Validates**: Gradient accumulation works correctly with device parallelism

---

### 4. `test_state_replication_unreplication`

**Purpose**: Verify state replication mechanics

**What it checks**:
1. **Replication**: Params get leading device axis
   - Unreplicated: `(voc_size, emb_dim)`
   - Replicated: `(num_devices, voc_size, emb_dim)`
2. **Identical copies**: All device copies have same values
3. **Unreplication**: Can recover original shape

**Code Example**:
```python
# Before replication
param.shape  # (128, 16)

# After replication to 2 devices
param.shape  # (2, 128, 16)

# Device copies are identical
assert jnp.allclose(param[0], param[1])
```

---

### 5. `test_checkpointing_multi_device`

**Purpose**: Test save/load with distributed state

**Validates**:
- Checkpoints save correctly with replicated state
- Loading restores and replicates properly
- State survives save/load cycle

---

## Running the Tests

### Run all distributed tests:
```bash
pytest tests/integration/test_distributed_training.py -v
```

### Run single test:
```bash
pytest tests/integration/test_distributed_training.py::TestDistributedTraining::test_distributed_overfitting_single_batch -v -s
```

### Expected output:
```
test_multi_cpu_device_setup PASSED
test_distributed_overfitting_single_batch PASSED
test_gradient_accumulation_multi_device PASSED
test_state_replication_unreplication PASSED
test_checkpointing_multi_device PASSED

5 passed in ~30s
```

---

## Implementation Details

### Why Use Subprocess?

XLA_FLAGS must be set **before** JAX imports. Tests use `subprocess.run()` to:
1. Set `XLA_FLAGS` in environment
2. Run test script in isolated Python process
3. JAX sees multiple devices on import

### Why Use CPU Simulation?

**Advantages**:
- ✅ Works in CI without GPU/TPU hardware
- ✅ Fast (< 30s for full suite)
- ✅ Deterministic and reliable
- ✅ Tests all distributed logic (pmap, replication, sharding)

**What it doesn't test**:
- GPU-specific performance optimizations
- TPU-specific pod slicing
- Multi-node distributed training
- GPU memory limits

But it **does** test the correctness of distributed training logic.

### Overfitting Strategy

**Why overfit instead of converge on real data?**

1. **Fast**: Takes 50 steps instead of 10,000
2. **Clear signal**: 99% loss reduction = definitely working
3. **No randomness**: Same batch every time, deterministic results
4. **Small data**: 4 sequences fits in memory easily

**The principle**: If the model can perfectly memorize a tiny batch, then:
- Forward pass works ✓
- Backward pass computes gradients correctly ✓
- pmap synchronizes gradients across devices ✓
- Optimizer updates all replicas consistently ✓

---

## Debugging Tips

### If test fails:

1. **Check XLA_FLAGS applied**:
   ```python
   import os
   print(os.environ.get('XLA_FLAGS'))  # Should have device count flag
   ```

2. **Verify devices created**:
   ```python
   import jax
   print(jax.local_device_count())  # Should be > 1
   ```

3. **Check loss trajectory**:
   - Not decreasing at all → forward/backward broken
   - Decreasing slowly → learning rate too low
   - Unstable (NaN) → learning rate too high

4. **Inspect replication**:
   ```python
   print(param.shape)  # Should have (num_devices, ...) for replicated
   ```

---

## Future Enhancements

Possible additions:
- [ ] Test with different device counts (3, 8, 16)
- [ ] Test data parallelism vs model parallelism
- [ ] Test mixed precision training
- [ ] Test gradient clipping with pmap
- [ ] Test evaluation mode with pmap
- [ ] Benchmark throughput vs single device

---

## Key Takeaways

1. **XLA_FLAGS simulates multi-device** without hardware
2. **Overfitting single batch** is gold standard for correctness
3. **All tests pass on CPU** - great for CI/CD
4. **~30s runtime** - fast feedback loop
5. **Comprehensive coverage** of distributed training logic

This test suite gives high confidence that distributed training works correctly before deploying to expensive GPU/TPU hardware.
