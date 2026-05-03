# OpenWebText Dataset

Pre-tokenized OpenWebText dataset for GPT-2 training.

## Quick Start

```bash
# Install kagglehub
pip install kagglehub

# Download dataset
python data/openwebtext/prepare.py
```

## Dataset Details

**Source:** [windmaple/openwebtext-gpt2](https://www.kaggle.com/datasets/windmaple/openwebtext-gpt2) on Kaggle  
**Size:** ~20GB total  
**Tokenizer:** GPT-2 BPE (vocab size 50,304)

**Files:**
- `train.bin` - Training set (~18GB, uint16 token IDs)
- `val.bin` - Validation set (~90MB, uint16 token IDs)

## Manual Download

If you prefer to download manually from Kaggle:

1. Download from: https://www.kaggle.com/datasets/windmaple/openwebtext-gpt2
2. Extract `train.bin` and `val.bin` to `data/openwebtext/`
3. Verify files exist:
   ```bash
   ls -lh data/openwebtext/
   # Should show train.bin (~18GB) and val.bin (~90MB)
   ```

## Usage in Training

Point your config to this directory:

```json
{
  "data": {
    "data_dir": "data/openwebtext",
    "train_file": "train.bin",
    "val_file": "val.bin"
  }
}
```

## Data Format

Both files are binary files containing uint16 token IDs:
- Each token is 2 bytes (uint16)
- Tokens are GPT-2 BPE encoding
- No special formatting or delimiters
- Read sequentially for training batches
