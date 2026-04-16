# Distributed Training Parallelism Seminar

Hands-on seminar on the three core parallelism strategies for training
neural networks across multiple GPUs: **data parallel**, **tensor
parallel**, and **pipeline parallel**.

## Requirements

- Node with 2+ NVIDIA GPUs
- Python 3.10+
- NCCL-capable PyTorch >= 2.4

```bash
python -m venv mlops
source mlops/bin/activate
pip install -r requirements.txt
```

## Configuration

All hyperparameters live in `conf/config.yaml` and can be overridden from
the CLI via Hydra:

```bash
./00_setup/launch.sh 2 02_data_parallel/train_ddp.py train.batch_size=32 model.n_layer=12
```

## Layout

| Directory / File | Topic | Key command |
|------------------|-------|-------------|
| `00_setup/` | Env check, shared utils, launcher | `jupyter notebook 00_setup/env_check.ipynb` |
| `01_baseline/` | Single-GPU reference numbers | `python 01_baseline/train.py` |
| `02_data_parallel/` | DDP and FSDP | `./00_setup/launch.sh 2 02_data_parallel/train_ddp.py` |
| `03_tensor_parallel/` | Megatron-style column/row parallel, DTensor | `./00_setup/launch.sh 2 03_tensor_parallel/train_tp_manual.py` |
| `04_pipeline_parallel/` | AFAB vs 1F1B schedules | `./00_setup/launch.sh 2 04_pipeline_parallel/train_pp_afab.py` |
| `SUMMARY.md` | Decision flowchart, strategy comparison | — |
| `metrics.csv` | Results from all parts (append-only) | `column -s, -t metrics.csv` |

Every part imports the same `TinyTransformer` from `01_baseline/model.py` —
students read only the parallelism diff, not a re-implementation of the model.

Each subdirectory has its own `README.md` with detailed instructions.
