# Summary — Choosing a parallelism strategy

## Decision flowchart

```
             Does the model + optimizer state fit on 1 GPU?
                        /                  \
                      yes                   no
                       |                     |
              Use single GPU         Does a single LAYER fit on 1 GPU?
                       |                  /              \
                Need more             yes                 no
              throughput?              |                    |
                /     \          Use DDP / FSDP        Add Tensor Parallel
              no      yes          |                   (intra-node, NVLink)
              |        |      Still need more               |
           done    Use DDP    throughput?            Still not enough?
                       |         /     \                /        \
                  Memory       no      yes            no         yes
                  tight?       |        |              |           |
                  /    \     done    Scale DP       DP x TP    Add Pipeline
                no     yes          (more nodes)   is enough   Parallel
                |       |                                         |
              DDP    FSDP                                    DP x TP x PP
                                                            (3D parallelism)
```

## Rules of thumb

| Strategy | When to use | Communication | Memory effect |
|----------|-------------|---------------|---------------|
| **DDP** | Model fits on 1 GPU, want more throughput | All-reduce gradients once per step | Same as single GPU |
| **FSDP** | Model fits on 1 GPU but memory is tight | All-gather params + reduce-scatter grads | Shards params/grads/optim across ranks |
| **Tensor Parallel** | A single layer is too large for 1 GPU | All-reduce per layer per fwd+bwd | Splits weight matrices across GPUs |
| **Pipeline Parallel** | Model has too many layers for TP alone | Point-to-point sends between stages | Each stage holds only its layers |
| **3D (DP x TP x PP)** | Very large models on many nodes | All of the above | All of the above |

## Placement guidelines

- **TP degree** <= GPUs per node (stay on NVLink — all-reduces every layer)
- **PP microbatches** >> num_stages (keep bubble fraction `(p-1)/(m+p-1)` small)
- **DP** is the outermost dimension: `world_size = DP x TP x PP`

## Results

After running the seminar scripts, compare all strategies in `metrics.csv`:

```bash
column -s, -t metrics.csv
```

## Further reading

- Megatron-LM (NVIDIA) — 3D parallelism for large language models
- DeepSpeed ZeRO (Microsoft) — memory-efficient data parallelism (ZeRO stages 1/2/3)
- PyTorch FSDP — built-in fully sharded data parallelism
- `torch.distributed.pipelining` — PyTorch-native pipeline parallelism
- Colossal-AI — unified parallelism framework
