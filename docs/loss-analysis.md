# Loss Analysis: Pretrained vs Untrained Models

This document establishes baseline loss targets for training janogpt from scratch.

## Evaluation Setup

- **Dataset**: OpenWebText (9B tokens)
- **Evaluation**: 10 batches × 4 sequences × 1024 tokens = 40,960 tokens
- **Metric**: Cross-entropy loss for next-token prediction
- **Hardware**: Evaluated on same hardware for fair comparison

## Results Summary

| Model | Parameters | Average Loss | Std Dev | Perplexity | vs Untrained |
|-------|-----------|--------------|---------|------------|--------------|
| **Untrained (random)** | 124M | **11.24** | ±0.02 | 76,308 | (baseline) |
| **GPT2 (pretrained)** | 124M | **3.18** | ±0.16 | 24 | **-8.06 (71% better)** |
| **GPT2-medium (pretrained)** | 354M | **2.90** | ±0.15 | 18 | **-8.34 (74% better)** |

**Theoretical maximum loss**: 10.82 (log(50257) - uniform distribution over vocabulary)

## Training Target

When training janogpt from scratch, expect this loss progression:

```
Initial (random):  ~11.2 loss  (perplexity ~76K)
                      ↓
                  [training]
                      ↓
Target (trained):  ~3.2 loss  (perplexity ~24)
```

**Success Criteria:**
- Loss should decrease from ~11 → ~3 over training
- Final loss ≤ 3.2 indicates GPT2-level performance
- Final loss ≤ 2.9 indicates GPT2-medium-level performance

## Detailed Results

### Untrained Model (Random Initialization)

```
Evaluating UNTRAINED GPT2 Model
Config: 12 blocks, 768 dim, 12 heads
Parameters: 124.4M

Batch-by-batch losses:
  Batch 1/10: loss = 11.2686
  Batch 2/10: loss = 11.2609
  Batch 3/10: loss = 11.2596
  Batch 4/10: loss = 11.2268
  Batch 5/10: loss = 11.2433
  Batch 6/10: loss = 11.2523
  Batch 7/10: loss = 11.1992
  Batch 8/10: loss = 11.2393
  Batch 9/10: loss = 11.2328
  Batch 10/10: loss = 11.2424

Average Loss: 11.2425 ± 0.0191
Perplexity:   76307.53
```

**Why is loss > theoretical max (10.82)?**

The theoretical maximum assumes a perfectly uniform distribution over the vocabulary. Random initialization produces a slightly non-uniform distribution, resulting in slightly higher loss. This is normal and expected.

### GPT2 (124M parameters, pretrained)

```
Evaluating gpt2
Config: 12 blocks, 768 dim, 12 heads
Parameters: 124.4M

Batch-by-batch losses:
  Batch 1/10: loss = 2.9994
  Batch 2/10: loss = 2.9659
  Batch 3/10: loss = 3.4303
  Batch 4/10: loss = 3.2650
  Batch 5/10: loss = 3.2559
  Batch 6/10: loss = 3.0382
  Batch 7/10: loss = 3.1690
  Batch 8/10: loss = 3.2633
  Batch 9/10: loss = 3.0309
  Batch 10/10: loss = 3.4124

Average Loss: 3.1830 ± 0.1604
Perplexity:   24.12
```

**Improvement over random**: -8.06 nats (71.7% reduction)

### GPT2-medium (354M parameters, pretrained)

```
Evaluating gpt2-medium
Config: 24 blocks, 1024 dim, 16 heads
Parameters: 354.8M

Batch-by-batch losses:
  Batch 1/10: loss = 2.7214
  Batch 2/10: loss = 2.6942
  Batch 3/10: loss = 3.0993
  Batch 4/10: loss = 2.9606
  Batch 5/10: loss = 2.9686
  Batch 6/10: loss = 2.7734
  Batch 7/10: loss = 2.8842
  Batch 8/10: loss = 3.0033
  Batch 9/10: loss = 2.7520
  Batch 10/10: loss = 3.1075

Average Loss: 2.8965 ± 0.1463
Perplexity:   18.11
```

**Improvement over random**: -8.35 nats (74.2% reduction)

**Improvement over GPT2**: -0.29 nats (9.0% reduction with 2.85× more parameters)

## Observations

### 1. Model Size vs Performance

Scaling from GPT2 (124M) to GPT2-medium (354M) gives:
- **2.85× more parameters**
- **Only 9% loss reduction** (3.18 → 2.90)

This demonstrates diminishing returns from model scaling alone.

### 2. Loss Variance

- **Untrained**: Very low variance (±0.02) - consistently random across batches
- **Pretrained**: Higher variance (±0.15) - some batches easier to predict than others

This variance reflects the inherent difficulty differences in natural text.

### 3. Perplexity Interpretation

- **Untrained (76K perplexity)**: Model is equally confused among all 50K+ tokens
- **GPT2 (24 perplexity)**: Model narrows choices to ~24 plausible tokens on average
- **GPT2-medium (18 perplexity)**: Model narrows to ~18 plausible tokens

Lower perplexity = more confident and accurate predictions.

## Training Expectations

When training janogpt from scratch, monitor these milestones:

| Training Progress | Expected Loss | Perplexity | Status |
|------------------|---------------|------------|--------|
| Step 0 (init) | ~11.2 | ~76K | Random baseline |
| Early training | ~8-10 | ~3K-20K | Learning basic patterns |
| Mid training | ~5-7 | ~150-1K | Learning structure |
| Late training | ~3.5-4.5 | ~30-90 | Approaching target |
| **Final (converged)** | **~3.2** | **~24** | **Target achieved** |

## How to Reproduce

Run the evaluation scripts:

```bash
# Evaluate untrained model
python << 'EOF'
from janogpt import Config, GPT
from janogpt.utils import FileDataLoader
# ... (see scripts/eval_pretrained_loss.py for full code)
EOF

# Evaluate pretrained models
python scripts/eval_pretrained_loss.py \
  --data_dir ../janogpt/data \
  --models gpt2 gpt2-medium \
  --num_batches 10 \
  --batch_size 4 \
  --seq_len 1024
```

## References

- **OpenWebText**: Open-source recreation of OpenAI's WebText dataset
- **GPT2 Paper**: [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- **Evaluation methodology**: Standard next-token prediction with cross-entropy loss

---

**Last Updated**: 2026-05-03

**Evaluation Hardware**: CPU (JAX)

**Data Source**: OpenWebText (9B tokens, train split used for evaluation)
