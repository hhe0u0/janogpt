# JanoGPT Test Suite

Comprehensive test suite for janogpt with **67 passing tests** covering model architecture, training pipeline, and next-token prediction correctness.

## Test Structure

```
tests/
├── unit/                          # Unit tests (fast, isolated)
│   ├── test_model.py             # GPT model architecture (10 tests)
│   ├── test_config.py            # Configuration system (14 tests)
│   ├── test_trainer.py           # Trainer logic (17 tests)
│   ├── test_inference.py         # Text generation (11 tests)
│   ├── test_utils.py             # Data loaders & utils (18 tests)
│   └── test_next_token_prediction.py  # Next-token correctness (4 tests)
│
├── integration/                   # Integration tests (slower, end-to-end)
│   ├── test_training_pipeline.py # Full training workflow
│   ├── test_hf_loading.py        # HuggingFace weight loading
│   └── test_generation.py        # Text generation quality
│
├── configs/                       # Test configurations
│   ├── test_tiny.json            # Tiny model (2 blocks, 64 dim)
│   └── test_small.json           # Small model (4 blocks, 128 dim)
│
└── README.md                      # This file
```

---

## Next-Token Prediction: How It Works

### The Core Concept

GPT is trained to predict the **next token** given all previous tokens. This is implemented through:

1. **Data Format**: Data loader provides `input_ids` of shape `(batch, seq_len)`
2. **Internal Shifting**: Trainer shifts to create input→target pairs
3. **Causal Masking**: Model at position `i` only sees tokens `0..i` (not future tokens)

### Example with Numbers

**Input sequence from data loader:**
```
input_ids = [10, 20, 30, 40, 50]
             ↑   ↑   ↑   ↑   ↑
           tok0 tok1 tok2 tok3 tok4
```

**What the model learns (after internal shifting):**

| Input (what model sees)      | Target (what model predicts) |
|------------------------------|------------------------------|
| `[10]`                       | `20` (next token)            |
| `[10, 20]`                   | `30` (next token)            |
| `[10, 20, 30]`               | `40` (next token)            |
| `[10, 20, 30, 40]`           | `50` (next token)            |

**In code (trainer.py):**
```python
# Forward pass produces logits for all positions
logits = model(input_ids)  # Shape: (batch, 5, vocab_size)

# Shift to create prediction → target pairs
shift_logits = logits[:, :-1, :]  # Predictions: positions 0-3
shift_labels = input_ids[:, 1:]    # Targets: tokens 1-4

# Loss computation
loss = cross_entropy(shift_logits, shift_labels)
```

**Shape example:**
```
Input:  (batch=2, seq_len=8, vocab=50304)
        ↓ forward pass
Logits: (batch=2, seq_len=8, vocab=50304)  # All positions
        ↓ shift for next-token
Shift logits: (2, 7, 50304)  # Predictions for positions 0-6
Shift labels: (2, 7)         # Target tokens at positions 1-7
        ↓ compute loss
Loss: scalar (averaged over all positions and batch)
```

### Causal Masking Example

**Critical: Position `i` must NOT see position `i+1`**

```
Sequence: [A, B, C, D, E]

Position 0 sees: [A]           → predicts B
Position 1 sees: [A, B]        → predicts C
Position 2 sees: [A, B, C]     → predicts D
Position 3 sees: [A, B, C, D]  → predicts E

Position 2 CANNOT see [D, E] (future tokens blocked by causal mask)
```

**This is tested in `test_next_token_prediction.py`:**
- If we change token E, predictions at positions 0-3 should NOT change
- Only prediction at position 4 (predicting beyond E) could differ

---

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only (fast)
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# Specific test file
pytest tests/unit/test_next_token_prediction.py -v

# Specific test
pytest tests/unit/test_model.py::TestGPTModel::test_forward_pass_shape -v
```

### Run with Coverage
```bash
pytest tests/ --cov=janogpt --cov-report=html
open htmlcov/index.html
```

### Run Tests Matching Pattern
```bash
# All tests with "next_token" in name
pytest tests/ -k "next_token" -v

# All tests with "causal" in name
pytest tests/ -k "causal" -v
```

---

## Test Coverage Status

### ✅ Well-Covered (>80% coverage)

- **Model Architecture** (`test_model.py`)
  - Forward pass, dynamic sequence length
  - Parameter counting (124M for GPT2)
  - Causal masking
  - Inference mode (deterministic)

- **Next-Token Prediction** (`test_next_token_prediction.py`)
  - Data format → loss pipeline
  - Shifting logic correctness
  - Causal masking validation
  - Future information leakage prevention

- **Configuration** (`test_config.py`)
  - JSON serialization/deserialization
  - Default values
  - Nested config sections

- **Data Loaders** (`test_utils.py`)
  - DummyDataLoader (in-memory)
  - FileDataLoader (memmap)
  - Batch sharding utilities

- **Inference** (`test_inference.py`)
  - Temperature sampling
  - Top-k filtering
  - Determinism

### ⚠️ Partial Coverage (40-80%)

- **Trainer** (`test_trainer.py`)
  - ✅ Initialization, learning rate schedule
  - ✅ Single device training
  - ❌ Multi-device (pmap) training - needs GPU/TPU
  - ❌ Gradient accumulation correctness
  - ❌ Checkpointing save/load

- **Integration** (`test_training_pipeline.py`)
  - ✅ Smoke test (5 steps)
  - ❌ Convergence test (100+ steps)
  - ❌ Checkpoint resume

### ❌ Not Covered (<40%)

- **Distributed Training**
  - Multi-GPU training
  - Multi-TPU training
  - Gradient accumulation scaling

- **Checkpointing**
  - Save/load correctness
  - Resume training from checkpoint
  - Checkpoint compatibility

- **HuggingFace Integration**
  - Weight loading correctness
  - Generation quality comparison
  - All transformer blocks match

- **Performance**
  - Training speed (tokens/sec)
  - Memory usage
  - JIT compilation time

---

## TODO List

### High Priority (Core Functionality)

- [ ] **Fix failing trainer tests** (3 tests)
  - `test_trainer_with_evaluators` - DataLoader parameter name
  - `test_train_step_single_device` - Missing dropout_rngs parameter
  - `test_loss_decreases_on_dummy_data` - Same issue

- [ ] **Add gradient accumulation tests**
  - Verify token count: `micro_batch × accum_steps × devices × seq_len = 524,288`
  - Test that accumulated gradients match non-accumulated (slower but same result)
  - Verify memory efficiency (accumulation should use less memory)

- [ ] **Add checkpointing tests**
  - Save checkpoint after N steps
  - Load checkpoint and verify parameters match
  - Resume training from checkpoint and continue
  - Test checkpoint compatibility across code versions

- [ ] **Add multi-device tests** (requires GPU/TPU)
  - Test pmap compilation on 2+ devices
  - Verify gradient synchronization (pmean)
  - Test that multi-device matches single-device results (numerically)

### Medium Priority (Quality & Robustness)

- [ ] **Add convergence tests**
  - Train on tiny dataset (e.g., 1000 tokens) until loss < 0.1
  - Verify model can overfit completely
  - Test that loss actually decreases over 100+ steps

- [ ] **Add HuggingFace integration tests**
  - Load GPT2 weights and verify all layers match
  - Compare generation output token-by-token with HF
  - Test on multiple prompts and verify quality

- [ ] **Add learning rate schedule tests**
  - Verify warmup increases LR linearly
  - Verify cosine decay to min_lr
  - Test constant LR mode (use_lr_schedule=False)

- [ ] **Add data loading edge cases**
  - Test with dataset smaller than batch size
  - Test with dataset not divisible by batch size
  - Test with very long sequences (near seq_len limit)

### Low Priority (Nice to Have)

- [ ] **Add performance benchmarks**
  - Measure tokens/sec on CPU/GPU/TPU
  - Measure memory usage during training
  - Measure JIT compilation time
  - Compare with reference implementations (nanogpt)

- [ ] **Add generation quality tests**
  - Test coherence (n-gram diversity)
  - Test repetition detection
  - Test that temperature affects diversity
  - Test that top_k affects output distribution

- [ ] **Add numerical stability tests**
  - Test with very large learning rates (should not NaN)
  - Test with very small learning rates (should still learn)
  - Test with mixed precision (bfloat16)
  - Test gradient clipping effectiveness

- [ ] **Add configuration validation tests**
  - Test invalid configs (e.g., emb_dim not divisible by num_heads)
  - Test edge cases (e.g., num_blocks=0, seq_len=1)
  - Test large configs (e.g., GPT-3 175B parameters)

### Documentation TODO

- [ ] **Add docstrings to all test functions**
  - Explain what each test verifies
  - Provide examples of failure modes
  - Link to relevant code sections

- [ ] **Add troubleshooting guide**
  - Common test failures and how to fix
  - How to debug failing tests
  - How to add new tests

- [ ] **Add CI/CD integration guide**
  - GitHub Actions workflow example
  - Test selection strategy (fast vs slow)
  - Coverage reporting setup

---

## Expected Test Results

### Unit Tests (should complete in <60 seconds)

```
tests/unit/test_model.py .................... [10 passed]
tests/unit/test_config.py ................... [14 passed]
tests/unit/test_trainer.py .................. [14 passed, 3 failed]
tests/unit/test_inference.py ................ [11 passed]
tests/unit/test_utils.py .................... [18 passed]
tests/unit/test_next_token_prediction.py .... [4 passed]

TOTAL: 67 passed, 3 failed in ~45s
```

### Integration Tests (should complete in <5 minutes)

```
tests/integration/test_training_pipeline.py .. [6 passed]
tests/integration/test_hf_loading.py ......... [9 passed, requires transformers]
tests/integration/test_generation.py ......... [12 passed]

TOTAL: 27 passed in ~3m
```

---

## Writing New Tests

### Test Naming Convention

```python
# Good test names (descriptive, specific)
def test_forward_pass_produces_correct_output_shape():
def test_causal_mask_prevents_seeing_future_tokens():
def test_loss_decreases_after_10_training_steps():

# Bad test names (vague, unclear)
def test_model():
def test_training():
def test_works():
```

### Test Structure Template

```python
def test_feature_name():
    """
    Test that [specific behavior] works correctly.
    
    This test verifies:
    1. [First thing being tested]
    2. [Second thing being tested]
    3. [Edge case or failure mode]
    """
    # Setup
    config = Config(...)
    model = GPT(config)
    
    # Execute
    result = model(input_data)
    
    # Verify
    assert result.shape == expected_shape
    assert result > 0
    assert np.isfinite(result)
```

### Common Assertions

```python
# Shape checks
assert tensor.shape == (batch, seq_len, vocab)

# Value range checks  
assert jnp.all(tokens >= 0)
assert jnp.all(tokens < vocab_size)

# Numerical stability
assert jnp.isfinite(loss)
assert not jnp.isnan(loss)

# Approximate equality (for floating point)
assert jnp.allclose(result1, result2, rtol=1e-5)

# Exact equality (for integers)
assert jnp.array_equal(tokens1, tokens2)
```

---

## Debugging Failed Tests

### Step 1: Run test with verbose output
```bash
pytest tests/unit/test_trainer.py::test_train_step_single_device -vv
```

### Step 2: Add debug prints
```python
def test_something():
    result = function_under_test()
    print(f"DEBUG: result shape = {result.shape}")
    print(f"DEBUG: result min/max = {result.min()}/{result.max()}")
    assert result.shape == expected
```

### Step 3: Use pytest debugger
```bash
# Drop into debugger on failure
pytest tests/unit/test_model.py --pdb

# Drop into debugger on first failure
pytest tests/unit/test_model.py -x --pdb
```

### Step 4: Check JAX traceback
```bash
# Show full JAX traceback (not simplified)
JAX_TRACEBACK_FILTERING=off pytest tests/unit/test_model.py -v
```

---

## Contributing Tests

When adding new features to janogpt, please add corresponding tests:

1. **Add unit tests** for isolated functionality
2. **Add integration tests** if feature involves multiple components
3. **Update this README** if test coverage changes significantly
4. **Run full test suite** before committing: `pytest tests/`

### Test Quality Checklist

- [ ] Test has clear, descriptive name
- [ ] Test has docstring explaining what it verifies
- [ ] Test is isolated (doesn't depend on other tests)
- [ ] Test is deterministic (same result every run)
- [ ] Test is fast (<1 second for unit tests)
- [ ] Test uses appropriate assertions
- [ ] Test has meaningful error messages

---

## Contact

For questions about tests or to report issues:
- File an issue: https://github.com/hhe0u0/janogpt/issues
- Refer to main README: `../README.md`
