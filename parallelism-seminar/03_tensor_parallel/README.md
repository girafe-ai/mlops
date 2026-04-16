# Part 3 — Tensor Parallelism

Split individual matmuls across GPUs, Megatron-LM style.

## What happens here

1. **Mini-lecture**: column-parallel and row-parallel Linear layers, the
   column->row pairing in the MLP, and why TP must stay intra-node (NVLink).
2. **Walk through** `tp_layers.py` — `ColumnParallelLinear` and
   `RowParallelLinear` using autograd-aware collectives from
   `torch.distributed.nn.functional`.
3. **Run** the model with the hand-written layers, then compare with
   the built-in `torch.distributed.tensor.parallel` API.

## Files

| File | Purpose |
|------|---------|
| `tp_layers.py` | `ColumnParallelLinear` / `RowParallelLinear` implementation |
| `train_tp_manual.py` | `TinyTransformer` using the hand-written parallel layers |
| `train_tp_dtensor.py` | Same model via `parallelize_module` + `ColwiseParallel` / `RowwiseParallel` |

## The column -> row pairing

A standard MLP is `fc2(gelu(fc1(x)))`. If `fc1` is column-parallel
(`gather_output=False`) and `fc2` is row-parallel (`input_is_parallel=True`),
the GELU runs on sharded activations with zero extra communication — only
one all-reduce per block instead of two.

## Commands

```bash
# Hand-written TP layers
./00_setup/launch.sh 2 03_tensor_parallel/train_tp_manual.py

# Built-in DTensor TP
./00_setup/launch.sh 2 03_tensor_parallel/train_tp_dtensor.py

# Override config
./00_setup/launch.sh 2 03_tensor_parallel/train_tp_manual.py train.batch_size=32
```

**Important:** TP degree must divide all sharded dimensions (`d_ff`, `d_model`).
With the default config (`d_ff=2048`), use 2 GPUs (not 3). In general, TP
degree is a power of 2.

TP should stay within a single node — the all-reduces run every forward
and backward and will saturate any link slower than NVLink.
