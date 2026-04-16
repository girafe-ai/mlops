# Part 0 — Environment & setup

Verify your multi-GPU node is ready before running anything.

## What happens here

- Check GPU count, NCCL availability, and inter-GPU topology (NVLink vs PCIe)
- Understand the distributed boilerplate: `RANK`, `WORLD_SIZE`, `LOCAL_RANK`
- Import `dist_utils` — the shared helper every later script relies on

## Files

| File | Purpose |
|------|---------|
| `env_check.ipynb` | Run the cells to verify GPUs, NCCL, topology |
| `dist_utils.py` | `setup_dist()` / `cleanup_dist()` / `is_main()` / `print_rank0()` |
| `conf_utils.py` | `load_config()` — loads `conf/config.yaml` via Hydra Compose API |
| `launch.sh` | Thin wrapper: `./launch.sh <NGPUS> <script.py> [overrides...]` |

## Commands

```bash
# Open the notebook
jupyter notebook env_check.ipynb

# Or just check from the terminal
python -c "import torch; print(torch.cuda.device_count())"
nvidia-smi topo -m
```
