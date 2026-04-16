# Part 1 — Single-GPU baseline

Train a small transformer on one GPU to establish reference numbers for
tokens/sec, peak memory, and step time. Every later part compares against these.

## What happens here

- Build `TinyTransformer` (~6 layers, d_model=512, ~13M params by default)
- Train on synthetic random token data for a few steps
- Log throughput and peak memory to `metrics.csv`

## Files

| File | Purpose |
|------|---------|
| `model.py` | `TinyTransformer` + `ModelConfig` — shared by all parts |
| `data.py` | `RandomTokenDataset` — deterministic synthetic data, no downloads |
| `train.py` | Single-GPU train loop, writes `metrics.csv` |

## Commands

```bash
cd 01_baseline

# Default config
python train.py

# Override parameters from CLI
python train.py train.batch_size=32 model.n_layer=12 train.steps=100
```

All parameters live in `conf/config.yaml` and can be overridden via CLI.
