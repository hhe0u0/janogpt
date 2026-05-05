# Debug Session Analysis: Training Infrastructure Issues

## Session Overview
**Duration**: ~2 days (78 commits)
**Primary Goal**: Train GPT-2 124M on TPU/GPU with gradient accumulation
**Outcome**: Multiple cascading issues requiring extensive debugging

---

## Issues Fixed (Chronological)

### 1. **Multi-Device Evaluation Shape Mismatch** 
**Error**: `ScopeParamShapeError: expected shape (50304, 768), got (2, 50304, 768)`

**Root Cause**: 
- Training replicates parameters across devices: `(num_devices, ...)`
- Evaluation expected single-device shape
- Evaluator called with replicated state

**Fix**: Add unreplication before evaluation
```python
eval_state = self.unreplicate(self.state) if self.num_devices > 1 else self.state
evaluator.evaluate(eval_state, ...)
```

**Prevention**:
- ✅ **Type annotations**: `state: ReplicatedState` vs `state: State`
- ✅ **Integration tests**: Test multi-device training + evaluation together
- ✅ **Assertion guards**: Check shape before evaluation

---

### 2. **Incorrect Memory Calculation (40× → 4×)**
**Error**: Memory estimate of ~5GB was wrong, should be ~2GB

**Root Cause**:
- Used incorrect formula: `40 × params` (where did 40 come from?)
- Should be: `4 × params` (params + grads + Adam m + Adam v)
- All in float32 for optimizer state

**Fix**: Corrected memory calculation formula in notebooks

**Prevention**:
- ✅ **Document formulas**: Add inline comments explaining calculations
- ✅ **Reference sources**: Link to paper/docs for memory formulas
- ✅ **Unit tests**: Test memory calculation function separately
- ✅ **Peer review**: Have calculations verified before committing

---

### 3. **Evaluation OOM on TPU v6e (32GB)**
**Error**: `RESOURCE_EXHAUSTED: Attempting to allocate 12.00G`

**Root Cause**:
- Training uses gradient accumulation: processes 1 micro-batch at a time
- Evaluation uses ENTIRE batch at once (no accumulation)
- `val_loader` used `effective_batch=512` → 12GB activation memory
- Training already using 22GB → Total 34GB > 32GB available

**Fix**: Use small eval batch size
```python
eval_batch_size = 8  # ~0.4GB instead of 12GB
val_loader = FileDataLoader(batch_size=eval_batch_size, ...)
```

**Prevention**:
- ✅ **Separate batch configs**: `train_batch` vs `eval_batch` from the start
- ✅ **Memory profiling**: Profile memory before full training
- ✅ **Smoke tests**: Small-scale test on target hardware first
- ✅ **Documentation**: Document memory requirements per configuration

---

### 4. **Missing JAX Imports**
**Error**: `name 'jax' is not defined`

**Root Cause**:
- `logger.py` used `jax.nn.softmax` but only imported `jax.numpy`
- `train_kaggle.ipynb` used `jax.local_device_count()` without import
- Inconsistent import patterns across files

**Fix**: Add `import jax` to affected files

**Prevention**:
- ✅ **Linting**: Use `ruff` or `flake8` to catch undefined names
- ✅ **CI checks**: Run static analysis on every commit
- ✅ **Import conventions**: Standardize imports in style guide
- ✅ **IDE setup**: Configure IDE to highlight missing imports

---

### 5. **Duplicate Evaluation Logs**
**Output**: 
```
[step 20] val/loss=8.6102 ...
[eval step=20] val: val/loss=8.6102 ...
```

**Root Cause**:
- Logger prints metrics via `logger.log()`
- Trainer ALSO prints same metrics manually
- No coordination between logging layers

**Fix**: Remove manual print statements, rely on logger

**Prevention**:
- ✅ **Single responsibility**: Logger handles ALL printing
- ✅ **Code review**: Check for duplicate logging
- ✅ **Logging framework**: Use structured logging (e.g., `loguru`)

---

### 6. **Checkpoint Loading - String Sorting Bug**
**Error**: Loads `step_50` when `step_100` exists

**Root Cause**:
- String sort: `"step_100" < "step_50"` (lexicographic: "1" < "5")
- Used `sorted(glob("step_*"))` without numeric comparison

**Fix**: Sort by numeric step value
```python
checkpoints = [(int(c.name.split("_")[1]), c) for c in ckpt_dir.glob("step_*")]
step = max(checkpoints)[0]
```

**Prevention**:
- ✅ **Test edge cases**: Test with steps 10, 20, 100, 1000
- ✅ **Unit tests**: Test checkpoint discovery logic
- ✅ **Semantic versioning**: Use zero-padded names: `step_00100`
- ⚠️  **Code review**: This is a classic Python gotcha

---

### 7. **Cleanup Deleting _train_step_fn**
**Error**: `AttributeError: 'Trainer' object has no attribute '_train_step_fn'`

**Root Cause**:
- First training run completes → `cleanup()` deletes `_train_step_fn`
- User tries to resume → Trainer object still exists but function is gone
- Training fails when trying to call deleted function

**Fix**: Don't manually delete attributes; let Python GC handle it
```python
def cleanup(self):
    jax.clear_caches()
    gc.collect()
    # Don't delete state or _train_step_fn!
```

**Prevention**:
- ✅ **Trust the GC**: Don't manually delete unless truly necessary
- ✅ **Test resumption**: Test save/load/resume in same process
- ✅ **Object lifecycle**: Document when objects should persist vs be cleaned
- ✅ **Context managers**: Use `with` blocks for automatic cleanup

---

### 8. **TrainState Restoration as Dict**
**Error**: `AttributeError: 'dict' object has no attribute 'params'`

**Root Cause**:
- Orbax's `PyTreeCheckpointer.restore()` returns plain dict
- Without target structure, it doesn't know to create TrainState object
- Code expected TrainState with `.params`, `.step` attributes

**Fix**: Provide target structure OR manually reconstruct
```python
# Option 1: Provide target
dummy_state = self.create_train_state()
restored = checkpointer.restore(path, item={"state": dummy_state, ...})

# Option 2: Manual reconstruction (chosen)
state_dict = restored["state"]
self.state = train_state.TrainState(
    step=state_dict["step"],
    params=state_dict["params"],
    ...
)
```

**Prevention**:
- ✅ **Test serialization**: Test save/load roundtrip early
- ✅ **Type checking**: Use `isinstance()` checks after deserialization
- ✅ **Documentation**: Document serialization format clearly
- ✅ **Abstraction**: Wrap checkpoint logic in dedicated class

---

### 9. **Optimizer State Structure Mismatch**
**Error**: `AttributeError: 'dict' object has no attribute 'mu'`

**Root Cause**:
- Orbax restores `opt_state` as nested dicts
- Optax expects NamedTuples with specific structure (`.mu`, `.nu` for Adam)
- Structure mismatch when trying to continue optimization

**Fix**: Recreate optimizer state fresh
```python
# Don't try to use restored opt_state
self.state = train_state.TrainState.create(
    apply_fn=self.model.apply,
    params=loaded_state["params"],
    tx=self.create_optimizer(),  # Fresh optimizer
)
```

**Tradeoff**: Lose momentum/variance state, but simpler and robust

**Prevention**:
- ✅ **Custom serializers**: Implement proper optax serialization
- ✅ **Version pinning**: Pin exact optax version for compatibility
- ✅ **Test framework changes**: Test after any dependency update
- ✅ **Accept tradeoffs**: Sometimes fresh state is acceptable

---

### 10. **State Access in Finally Block**
**Error**: `AttributeError: 'Trainer' object has no attribute 'state'`

**Root Cause**:
- `finally` block runs even if exception occurs in `__init__`
- If initialization fails before `self.state` is set, finally block tries to access it
- No guard for missing attributes

**Fix**: Check attribute exists before accessing
```python
finally:
    if hasattr(self, 'state'):
        # Run final evaluation and checkpoint
        ...
```

**Prevention**:
- ✅ **Defensive programming**: Always check `hasattr()` in cleanup code
- ✅ **Initialization**: Set attributes to None in `__init__` early
- ✅ **Separate concerns**: Don't put complex logic in finally blocks
- ✅ **Context managers**: Use try/finally only for cleanup, not business logic

---

### 11. **Cache Memory Accumulation**
**Error**: `RESOURCE_EXHAUSTED: Out of memory` during final evaluation

**Root Cause**:
- JAX compilation cache accumulates during training
- Final evaluation runs with cache still in memory
- Combined memory exceeds available 32GB on TPU

**Fix**: Clear caches before final evaluation
```python
finally:
    if hasattr(self, 'state'):
        print("Training complete! Running final evaluation...")
        jax.clear_caches()  # Free compilation cache
        ...
```

**Prevention**:
- ✅ **Periodic clearing**: Clear caches periodically during training
- ✅ **Memory monitoring**: Log memory usage throughout training
- ✅ **Lazy evaluation**: Only compile functions when needed
- ✅ **Cache limits**: Configure JAX cache size limits

---

## Root Cause Categories

### 1. **Insufficient Testing** (50% of issues)
Issues that would have been caught by proper tests:
- Multi-device evaluation shape mismatch
- String sorting bug
- Cleanup deleting functions
- TrainState restoration
- Optimizer state mismatch
- State access in finally block

**Missing Test Coverage**:
- ❌ Multi-device training + evaluation together
- ❌ Checkpoint save/load/resume in same process
- ❌ Edge cases (step_100 vs step_50)
- ❌ Serialization roundtrips

---

### 2. **Resource Management** (25% of issues)
Issues related to memory/resource handling:
- Evaluation OOM (wrong batch size)
- Cache memory accumulation
- Incorrect memory calculation

**Missing Infrastructure**:
- ❌ Memory profiling before full training
- ❌ Resource monitoring during execution
- ❌ Smoke tests on target hardware

---

### 3. **Code Quality** (15% of issues)
Issues from poor code organization:
- Missing imports
- Duplicate logging
- Manual resource deletion

**Missing Practices**:
- ❌ Static analysis (linting)
- ❌ Import conventions
- ❌ Single responsibility principle

---

### 4. **Documentation Gaps** (10% of issues)
Issues from lack of documentation:
- Incorrect memory formula (no reference)
- Unclear checkpoint path handling

**Missing Documentation**:
- ❌ Memory calculation formulas with sources
- ❌ API contracts for state replication
- ❌ Checkpoint format specification

---

## Time Waste Analysis

**Total Commits**: 78 commits over ~2 days
**Estimated Time**: ~16-20 hours of debugging

### Time Breakdown:
1. **Checkpoint issues**: ~8 commits, ~3 hours
   - String sorting, path handling, TrainState restoration, opt_state
2. **Memory/OOM issues**: ~5 commits, ~2 hours
   - Memory calculation, eval batch size, cache clearing
3. **Multi-device issues**: ~3 commits, ~1.5 hours
   - Shape mismatch, unreplication
4. **Code quality**: ~4 commits, ~1 hour
   - Missing imports, duplicate logs
5. **Infrastructure**: ~8 commits, ~2 hours
   - Notebooks, configs, dynamic calculations

**Avoidable with Tests**: ~60% (12 hours)
**Avoidable with Better Design**: ~30% (6 hours)
**Legitimate Discovery**: ~10% (2 hours)

---

## Prevention Strategies (Prescriptive)

### Immediate Actions (Week 1)

#### 1. **Add Comprehensive Tests**
```bash
# Create test suite
tests/
  test_trainer.py          # Unit tests for Trainer
  test_checkpoint.py       # Checkpoint save/load
  test_multi_device.py     # Multi-device training
  test_memory.py           # Memory calculations
  test_integration.py      # End-to-end workflows
```

**Priority Tests**:
- ✅ Multi-device training → evaluation (replicated state)
- ✅ Checkpoint save → load → resume (same process)
- ✅ Edge cases: step_10 vs step_100 sorting
- ✅ Memory estimation vs actual usage
- ✅ Serialization roundtrips

#### 2. **Add Static Analysis**
```bash
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff           # Fast linter
      - id: ruff-format    # Fast formatter
  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy           # Type checking
```

#### 3. **Add CI Pipeline**
```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -e .[dev]
      - name: Run linters
        run: ruff check .
      - name: Run type checker
        run: mypy janogpt/
      - name: Run tests
        run: pytest tests/ -v
```

---

### Short-term Improvements (Month 1)

#### 4. **Improve Type Safety**
```python
# janogpt/types.py
from typing import TypeAlias, Protocol
import jax.numpy as jnp

# Distinguish single-device vs multi-device state
State: TypeAlias = train_state.TrainState
ReplicatedState: TypeAlias = train_state.TrainState  # Shape: (num_devices, ...)

class Evaluator(Protocol):
    def evaluate(self, state: State, ...) -> dict[str, float]:
        """Evaluator expects UNREPLICATED state."""
        ...
```

#### 5. **Add Resource Monitoring**
```python
# janogpt/monitor.py
class ResourceMonitor:
    def __init__(self, device_type: str):
        self.device_type = device_type
    
    def log_memory(self, step: int):
        """Log current memory usage."""
        if self.device_type == "tpu":
            # Log TPU HBM usage
            ...
        elif self.device_type == "gpu":
            # Log GPU memory
            ...
    
    def check_memory_headroom(self, required_gb: float):
        """Verify sufficient memory before operation."""
        available = self.get_available_memory()
        if available < required_gb:
            raise MemoryError(f"Need {required_gb}GB, only {available}GB available")
```

#### 6. **Checkpoint Abstraction**
```python
# janogpt/checkpoint.py
class CheckpointManager:
    """Handles all checkpoint save/load logic."""
    
    def save(self, state: State, step: int, path: Path):
        """Save checkpoint with proper serialization."""
        ...
    
    def load(self, path: Path, model: nn.Module) -> tuple[State, int]:
        """Load checkpoint with proper TrainState reconstruction."""
        ...
    
    def list_checkpoints(self, dir: Path) -> list[tuple[int, Path]]:
        """List checkpoints sorted by step number (numeric)."""
        checkpoints = dir.glob("step_*")
        return sorted([(int(p.name.split("_")[1]), p) for p in checkpoints])
    
    def find_latest(self, dir: Path) -> Path:
        """Find latest checkpoint."""
        checkpoints = self.list_checkpoints(dir)
        if not checkpoints:
            raise ValueError(f"No checkpoints in {dir}")
        return checkpoints[-1][1]
```

---

### Long-term Improvements (Quarter 1)

#### 7. **Memory Profiling Integration**
```python
# Before training
profiler = MemoryProfiler(device_type="tpu")
profiler.estimate_memory(config)
profiler.verify_fits(max_memory_gb=32)

# During training
with profiler.track("training_step"):
    state, metrics = train_step(state, batch)

profiler.report()  # Show memory timeline
```

#### 8. **Smoke Test Framework**
```python
# scripts/smoke_test.py
def smoke_test_training(config: Config, device_type: str):
    """Run quick training test on target hardware."""
    # Override config for fast test
    config.max_steps = 3
    config.eval_interval = 2
    config.save_interval = 2
    
    # Run full pipeline
    trainer = Trainer(model, config, ...)
    trainer.train(train_loader)
    
    # Verify checkpoint works
    trainer2 = Trainer(model, config.with_resume(...), ...)
    trainer2.train(train_loader)
    
    print("✓ Smoke test passed on {device_type}")
```

#### 9. **Better Error Messages**
```python
# Before
raise ValueError(f"No checkpoints found in {ckpt_dir}")

# After
raise ValueError(
    f"No checkpoints found in {ckpt_dir}\n"
    f"Expected format: step_<N>/ (e.g., step_100/)\n"
    f"Available files: {list(ckpt_dir.glob('*'))}"
)
```

#### 10. **Documentation Standards**
```python
class Trainer:
    def unreplicate(self, state: ReplicatedState) -> State:
        """
        Convert replicated state to single-device state.
        
        Multi-device training replicates parameters across devices with shape:
            (num_devices, ...)
        
        This function extracts the first replica for single-device operations
        like evaluation or checkpointing.
        
        Args:
            state: Replicated TrainState with shape (num_devices, ...)
        
        Returns:
            Single-device TrainState with shape (...)
        
        Example:
            >>> # Training state (2 GPUs)
            >>> state.params['Dense_0']['kernel'].shape  # (2, 768, 3072)
            >>> unreplicated = trainer.unreplicate(state)
            >>> unreplicated.params['Dense_0']['kernel'].shape  # (768, 3072)
        
        Warning:
            Only call this when num_devices > 1. Single-device state is
            already in the correct format.
        """
        ...
```

---

## Cost-Benefit Analysis

### Testing Infrastructure
**Effort**: 2-3 days to set up comprehensive tests
**Benefit**: Would have caught 60% of issues (12 hours saved)
**ROI**: 4-6x time savings on this session alone
**Ongoing**: Prevents future regressions

### Static Analysis
**Effort**: 1 day to configure linters, type checker, pre-commit hooks
**Benefit**: Catches 15% of issues automatically (3 hours saved)
**ROI**: 3x time savings, plus ongoing quality improvement
**Ongoing**: Free automated checks

### Better Abstractions
**Effort**: 2-3 days to refactor (CheckpointManager, ResourceMonitor)
**Benefit**: Reduces complexity, easier to test, fewer bugs
**ROI**: 2-3x in future feature development
**Ongoing**: Easier maintenance and extensions

### Documentation
**Effort**: 1 day to document critical functions and formulas
**Benefit**: Reduces confusion, easier onboarding
**ROI**: 2x for teams, moderate for solo
**Ongoing**: Reference for future decisions

---

## Recommendations (Prioritized)

### Priority 1 (Do First - Highest ROI)
1. ✅ **Add test_checkpoint.py and test_training_e2e.py** (Done!)
2. ⚠️  Add multi-device integration test
3. ⚠️  Set up pre-commit hooks with ruff
4. ⚠️  Add CI pipeline with pytest

### Priority 2 (Do This Week)
5. Refactor checkpoint logic into CheckpointManager class
6. Add memory monitoring and headroom checks
7. Add smoke test script for new hardware
8. Document memory formulas with references

### Priority 3 (Do This Month)
9. Add type annotations and mypy checking
10. Improve error messages with actionable hints
11. Add resource profiling integration
12. Write contribution guide with standards

---

## Key Lessons

### 1. **Test Early, Test Often**
> "12 hours of debugging could have been 3 hours of writing tests"

The lack of tests for checkpoint save/load/resume was the single biggest time sink. Tests would have caught issues immediately.

### 2. **Don't Fight the Framework**
> "Trying to restore optax state from dicts was a losing battle"

Sometimes the simple solution (recreate fresh state) is better than the "correct" solution (properly deserialize structures).

### 3. **Resource Management is Hard**
> "Training and evaluation have different memory profiles"

Batch sizes that work for training don't work for evaluation. Memory profiling should be built-in, not an afterthought.

### 4. **Manual Cleanup is Dangerous**
> "Deleting _train_step_fn broke resumption"

Trust Python's garbage collector. Manual deletion should be rare and well-justified.

### 5. **Test the Integration, Not Just Units**
> "Each piece worked alone, but not together"

Multi-device training + evaluation integration wasn't tested. End-to-end tests would have caught it.

---

## Conclusion

**Total Issues**: 11 major bugs fixed
**Total Time**: ~16-20 hours debugging
**Preventable**: ~12-15 hours (60-75%)

**Primary Prevention**: Comprehensive testing
**Secondary Prevention**: Static analysis and type checking
**Tertiary Prevention**: Better abstractions and documentation

**Next Steps**:
1. ✅ Tests added (checkpoint, e2e)
2. ⏳ Add remaining test coverage (multi-device, memory)
3. ⏳ Set up pre-commit hooks and CI
4. ⏳ Refactor checkpoint logic
5. ⏳ Add resource monitoring

The debugging session revealed systemic issues in testing and code organization. Addressing these will prevent similar cascading failures in the future.
