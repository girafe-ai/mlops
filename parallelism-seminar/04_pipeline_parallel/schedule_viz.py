"""Pure-Python simulator + Gantt chart for pipeline schedules.

No GPUs required — this is a visual aid. Run:

    python schedule_viz.py --stages 4 --microbatches 8 --schedule afab
    python schedule_viz.py --stages 4 --microbatches 8 --schedule 1f1b

Each stage takes 1 unit for forward and 1 unit for backward.
"""

import argparse
from dataclasses import dataclass

import matplotlib.patches as patches
import matplotlib.pyplot as plt


@dataclass
class Op:
    stage: int
    microbatch: int
    kind: str  # "F" or "B"
    start: float
    duration: float = 1.0


def simulate_afab(num_stages: int, num_microbatches: int) -> list[Op]:
    """All forwards then all backwards. Big bubble, simple to reason about."""
    ops: list[Op] = []
    # Forward: microbatch m becomes available at stage s at time m + s.
    for m in range(num_microbatches):
        for s in range(num_stages):
            ops.append(Op(stage=s, microbatch=m, kind="F", start=float(m + s)))
    # Backward: reverse direction, starts after the last forward of the
    # final stage finishes.
    fwd_end = num_microbatches + num_stages - 1
    for m in range(num_microbatches):
        for i, s in enumerate(reversed(range(num_stages))):
            ops.append(
                Op(stage=s, microbatch=m, kind="B", start=float(fwd_end + m + i))
            )
    return ops


def simulate_1f1b(num_stages: int, num_microbatches: int) -> list[Op]:
    """1F1B steady-state schedule (PipeDream-Flush / Megatron default).

    Warm-up: each stage `s` does `(num_stages - s)` forwards.
    Steady state: alternate 1 forward / 1 backward per step.
    Cool-down: each stage drains its pending backwards.
    """
    ops: list[Op] = []
    # Track next free time per stage and the queue of microbatches in flight.
    stage_time = [0.0] * num_stages
    # Per-stage queue of microbatches that have done forward but not backward.
    in_flight: list[list[int]] = [[] for _ in range(num_stages)]
    # Per-stage counter for the next microbatch index we will forward.
    next_fwd = [0] * num_stages

    def do_fwd(s: int):
        m = next_fwd[s]
        next_fwd[s] += 1
        # A stage-s forward cannot start before the previous stage finished
        # forwarding the same microbatch.
        dep = 0.0 if s == 0 else _last_op_end(ops, s - 1, m, "F")
        start = max(stage_time[s], dep)
        ops.append(Op(stage=s, microbatch=m, kind="F", start=start))
        stage_time[s] = start + 1.0
        in_flight[s].append(m)

    def do_bwd(s: int):
        m = in_flight[s].pop(0)
        # Backward at stage s needs the backward of stage s+1 for the same mb.
        dep = 0.0 if s == num_stages - 1 else _last_op_end(ops, s + 1, m, "B")
        start = max(stage_time[s], dep)
        ops.append(Op(stage=s, microbatch=m, kind="B", start=start))
        stage_time[s] = start + 1.0

    # Schedule stage by stage in lockstep rounds so dependencies resolve.
    warmup = [num_stages - s for s in range(num_stages)]
    for s in range(num_stages):
        for _ in range(min(warmup[s], num_microbatches)):
            do_fwd(s)

    steady = [max(0, num_microbatches - warmup[s]) for s in range(num_stages)]
    for _ in range(max(steady)):
        for s in range(num_stages):
            if steady[s] > 0:
                do_fwd(s)
                do_bwd(s)
                steady[s] -= 1
            elif in_flight[s]:
                do_bwd(s)

    # Drain remaining backwards.
    while any(in_flight):
        for s in range(num_stages):
            if in_flight[s]:
                do_bwd(s)
    return ops


def _last_op_end(ops: list[Op], stage: int, microbatch: int, kind: str) -> float:
    for op in reversed(ops):
        if op.stage == stage and op.microbatch == microbatch and op.kind == kind:
            return op.start + op.duration
    return 0.0


def plot(ops: list[Op], num_stages: int, title: str):
    fig, ax = plt.subplots(figsize=(10, 1.2 * num_stages + 0.5))
    colors = {"F": "#4c9be8", "B": "#e88a4c"}
    for op in ops:
        rect = patches.Rectangle(
            (op.start, num_stages - 1 - op.stage - 0.4),
            op.duration,
            0.8,
            edgecolor="black",
            facecolor=colors[op.kind],
        )
        ax.add_patch(rect)
        ax.text(
            op.start + op.duration / 2,
            num_stages - 1 - op.stage,
            f"{op.kind}{op.microbatch}",
            ha="center",
            va="center",
            fontsize=8,
        )
    end = max(op.start + op.duration for op in ops)
    ax.set_xlim(0, end)
    ax.set_ylim(-0.5, num_stages - 0.5)
    ax.set_yticks(range(num_stages))
    ax.set_yticklabels([f"stage {num_stages - 1 - i}" for i in range(num_stages)])
    ax.set_xlabel("time")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=int, default=4)
    ap.add_argument("--microbatches", type=int, default=8)
    ap.add_argument("--schedule", choices=["afab", "1f1b"], default="afab")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    sim = simulate_afab if args.schedule == "afab" else simulate_1f1b
    ops = sim(args.stages, args.microbatches)

    makespan = max(op.start + op.duration for op in ops)
    ideal = 2 * args.microbatches  # 1 fwd + 1 bwd per mb on an infinite pipeline
    bubble = 1 - ideal / makespan
    print(f"schedule   : {args.schedule}")
    print(f"stages     : {args.stages}")
    print(f"microbatch : {args.microbatches}")
    print(f"makespan   : {makespan:.0f}")
    print(f"bubble frac: {bubble:.2%}")

    fig = plot(
        ops, args.stages, f"{args.schedule} — p={args.stages} m={args.microbatches}"
    )
    if args.out:
        fig.savefig(args.out, dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()
