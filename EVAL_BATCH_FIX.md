# Evaluation Batch Size Fix

## Problem

Training OOM during evaluation with error:
```
RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 12.00G
```

## Root Cause

**Evaluation uses the SAME batch size as training!**

```python
effective_batch = micro_batch × accum_steps × num_devices
                = 1 × 512 × 1 = 512 sequences

val_loader = FileDataLoader(batch_size=effective_batch)  # ← TOO LARGE!
```

With 512 sequences × 434MB activations = **12GB just for evaluation!**

But training already uses ~22GB, so:
- Training: ~22 GB
- Evaluation: +12 GB
- Total: 34 GB > 32GB TPU memory ❌

## Solution

**Use separate small batch for evaluation:**

```python
# Calculate training batch (can be large with gradient accumulation)
effective_batch = config.micro_batch_size * config.gradient_accumulation_steps * num_devices

# Training loader
train_loader = FileDataLoader(
    data_dir=config.data_dir,
    batch_size=effective_batch,  # Large batch OK (processed incrementally)
    seq_len=config.seq_len,
    split="train",
    seed=config.seed,
)

# IMPORTANT: Use SMALL batch for evaluation
eval_batch_size = 8  # Fits in ~0.4GB (vs 12GB with batch=512)

val_loader = FileDataLoader(
    data_dir=config.data_dir,
    batch_size=eval_batch_size,  # ← SMALL batch!
    seq_len=config.seq_len,
    split="val",
    seed=config.seed + 1,
)
```

## Memory Breakdown

**With eval_batch_size=8:**
- Parameters + optimizer: ~2 GB
- Training activations (micro_batch=1): ~0.4 GB  
- Evaluation activations (batch=8): ~3.5 GB
- **Total: ~6 GB** ✅ Fits in 32GB!

**Previous (eval_batch_size=512):**
- Parameters + optimizer: ~2 GB
- Training state: ~20 GB
- Evaluation activations (batch=512): ~12 GB
- **Total: ~34 GB** ❌ OOM!

## Files to Update

1. ✅ `train_colab.ipynb` - cell 20 (training setup)
2. ✅ `train_kaggle_tpu.ipynb` - cell 20 (training setup)  
3. ✅ `train_kaggle.ipynb` - cell 16 (training setup)

## Why This Happens

During gradient accumulation:
- Training: Processes 1 micro-batch at a time, accumulates gradients
- Evaluation: Tries to process ENTIRE batch at once (no accumulation)

So evaluation needs **much smaller batches** than training!
