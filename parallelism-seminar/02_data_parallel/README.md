# Part 2 — Data Parallelism (DDP & FSDP)

Replicate the model across GPUs, shard the batch, all-reduce gradients.

## What happens here

1. **DDP** (`train_ddp.py`): each GPU holds a **full copy** of the model
   (parameters, gradients, optimizer states). The batch is sharded across
   GPUs. After the backward pass, gradients are all-reduced so every replica
   stays in sync. Communication volume: `2 * (N-1)/N * |params|`.

2. **FSDP** (`train_fsdp.py`): the batch is still sharded the same way as
   DDP, but now **parameters, gradients, and optimizer states are also
   sharded** across ranks. Before each layer's forward pass, FSDP all-gathers
   that layer's parameters; after backward, it reduce-scatters the gradients
   and discards the full parameters. Each GPU only stores `1/N` of the model
   at rest. Same math, same convergence — FSDP just trades extra
   communication for lower per-GPU memory.

3. Discuss common pitfalls (see below).

## DDP vs FSDP — when does FSDP actually help?

FSDP saves memory by sharding three things: params, gradients, and optimizer
states (AdamW keeps two extra tensors per parameter: momentum + variance).
For a model with `P` parameters in fp32:

| Component | Per-GPU with DDP | Per-GPU with FSDP (N GPUs) |
|-----------|-----------------|---------------------------|
| Parameters | `4P` bytes | `4P / N` bytes |
| Gradients | `4P` bytes | `4P / N` bytes |
| Optimizer (AdamW) | `8P` bytes | `8P / N` bytes |
| **Total model state** | **`16P` bytes** | **`16P / N` bytes** |

With our default `TinyTransformer` (~27M params), model state is only ~430 MB
— memory is dominated by **activations** (batch_size × block_size × d_model ×
num_layers), so sharding the model barely matters.

FSDP becomes essential when the model is large enough that the 16P bytes
dominate memory. To see the difference in the seminar, use a bigger model
with a small batch:

```bash
# Compare DDP vs FSDP with a larger model — watch peak memory
./00_setup/launch.sh 3 02_data_parallel/train_ddp.py model.n_layer=24 model.d_model=1024 train.batch_size=8
./00_setup/launch.sh 3 02_data_parallel/train_fsdp.py model.n_layer=24 model.d_model=1024 train.batch_size=8
```

## FSDP and DeepSpeed ZeRO

FSDP is PyTorch's built-in equivalent of DeepSpeed ZeRO. The mapping:

| DeepSpeed ZeRO | What's sharded | PyTorch equivalent |
|----------------|----------------|--------------------|
| Stage 1 | optimizer states only | — (no built-in equivalent) |
| Stage 2 | optimizer states + gradients | — |
| Stage 3 | optimizer states + gradients + parameters | **FSDP** (default) |

FSDP always does the most aggressive sharding (like ZeRO Stage 3). DeepSpeed
gives finer control — Stage 1 or 2 have less communication overhead but save
less memory. In practice, if FSDP is too slow for your model, consider
DeepSpeed ZeRO Stage 2 as a middle ground.

## Files

| File | Purpose |
|------|---------|
| `train_ddp.py` | DDP version of the baseline |
| `train_fsdp.py` | FSDP version with `transformer_auto_wrap_policy` |
| `run.sh` | Launches both at world sizes 2 and 4 |

## Commands

```bash
# DDP on 2 GPUs
./00_setup/launch.sh 2 02_data_parallel/train_ddp.py

# DDP on 3 GPUs with bigger batch
./00_setup/launch.sh 3 02_data_parallel/train_ddp.py train.batch_size=32

# FSDP on 2 GPUs
./00_setup/launch.sh 2 02_data_parallel/train_fsdp.py

# Run all combinations at once
cd 02_data_parallel && ./run.sh
```

## Common pitfalls

### 1. Forgetting `sampler.set_epoch(epoch)`

`DistributedSampler` seeds its shuffle from the epoch number. Without
`set_epoch`, every epoch sees the same order on every rank.

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)   # required
    for x, y in loader:
        ...
```

### 2. Logging from every rank

```python
print(f"step {step}  loss {loss.item():.3f}")   # 3 GPUs = 3x spam
```

Fix: use `print_rank0()` from `dist_utils`.

### 3. Saving `model` instead of `model.module`

DDP wraps the model. `model.state_dict()` adds a `module.` prefix to
every key, which breaks loading into the unwrapped architecture.

```python
# wrong
torch.save(model.state_dict(), "ckpt.pt")

# right — unwrap first
torch.save(model.module.state_dict(), "ckpt.pt")
```

For FSDP, use `FSDP.state_dict_type(...)` to control sharding on save.
