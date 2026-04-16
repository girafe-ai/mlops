# Part 4 — Pipeline Parallelism

Split the model by layers across stages and stream microbatches.

## What happens here

The model is divided into **p stages** (e.g. first half of layers on GPU 0,
second half on GPU 1). A training batch is split into **m micro-batches**
that flow through the stages one after another.

The problem: GPUs can't all be busy at the same time — some sit idle waiting
for input from the previous stage or gradients from the next. This idle time
is called the **bubble**.

### Scheduling strategies

**Naive schedule** — stage 0 processes micro-batch 0 fully (forward +
backward), then micro-batch 1, etc. Stages 1, 2, 3 sit idle until their
turn. Almost all time is bubble.

**AFAB (all-forward-all-backward)** — all m micro-batches go forward through
all stages first, then all go backward. Better than naive, but still has a
big bubble at the start (stages wait for the first micro-batch to reach them)
and in the middle (between forward and backward phases). AFAB also stores
activations for all m micro-batches, so memory grows with m.

**1F1B (one-forward-one-backward)** — after a short warm-up, each stage
alternates: 1 forward, 1 backward, 1 forward, 1 backward, ... This reaches
a steady state faster, and each stage only holds activations for ~p
micro-batches in memory (vs m for AFAB). Memory is flat regardless of how
many micro-batches you use.

### The bubble formula

```
bubble fraction = (p - 1) / (m + p - 1)
```

- **p** = number of stages (GPUs in the pipeline)
- **m** = number of micro-batches the batch is split into

Example: p=4, m=4 → bubble = 3/7 = **43%** (almost half the time wasted).
With m=16 → bubble = 3/19 = **16%**. With m=32 → 3/35 = **9%**.

That's why the recommendation is **m >> p** — more micro-batches push the
bubble toward zero. The `bubble_experiment.py` script confirms this
numerically.

## Files

| File | Purpose |
|------|---------|
| `schedule_viz.py` | CPU-only AFAB/1F1B simulator + matplotlib Gantt chart |
| `bubble_experiment.py` | Sweeps microbatch counts, compares measured bubble to the formula |
| `train_pp_afab.py` | 2-stage split with `ScheduleGPipe` (AFAB schedule) |
| `train_pp_1f1b.py` | Same split with `Schedule1F1B` |

## Commands

```bash
# Visualize schedules (no GPU needed)
python 04_pipeline_parallel/schedule_viz.py --stages 4 --microbatches 4 --schedule afab
python 04_pipeline_parallel/schedule_viz.py --stages 4 --microbatches 4 --schedule 1f1b

# See the bubble collapse as m grows
python 04_pipeline_parallel/bubble_experiment.py

# AFAB on 2 GPUs
./00_setup/launch.sh 2 04_pipeline_parallel/train_pp_afab.py

# 1F1B on 2 GPUs
./00_setup/launch.sh 2 04_pipeline_parallel/train_pp_1f1b.py

# More microbatches (batch_size must be divisible by num_microbatches)
./00_setup/launch.sh 2 04_pipeline_parallel/train_pp_afab.py pipeline.num_microbatches=8 train.batch_size=32
```

## Gotchas

- `PipelineStage` needs an `input_args` example tensor matching the shape
  and dtype the stage receives. Stage 0 gets `long` token IDs; later stages
  get float activations.
- Only the last stage produces the loss — gate printing accordingly.
- `train.batch_size` must be divisible by `pipeline.num_microbatches`.
