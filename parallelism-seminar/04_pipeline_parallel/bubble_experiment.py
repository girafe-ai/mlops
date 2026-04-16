"""Sweep the number of microbatches and plot bubble fraction vs throughput.

No GPUs required — we reuse the Python simulator from schedule_viz.py to
compute the theoretical bubble and compare it against the formula
``(p - 1) / (m + p - 1)``. The intent is to let students build intuition
for why ``m >> p`` is the usual recommendation before they launch the
real PP training scripts.

Run:
    python bubble_experiment.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from schedule_viz import simulate_1f1b, simulate_afab


def bubble_fraction(ops, num_microbatches: int) -> float:
    makespan = max(op.start + op.duration for op in ops)
    ideal = 2 * num_microbatches
    return 1 - ideal / makespan


def theoretical(p: int, m: int) -> float:
    return (p - 1) / (m + p - 1)


def main():
    stages = 4
    ms = [1, 2, 4, 8, 16, 32]

    rows = []
    for m in ms:
        gp = bubble_fraction(simulate_afab(stages, m), m)
        f1 = bubble_fraction(simulate_1f1b(stages, m), m)
        th = theoretical(stages, m)
        rows.append((m, gp, f1, th))
        print(f"m={m:3d}  afab={gp:.2%}  1f1b={f1:.2%}  theory={th:.2%}")

    out_csv = Path(__file__).parent / "bubble.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["microbatches", "afab_bubble", "1f1b_bubble", "theory"])
        w.writerows(rows)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", label="AFAB")
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "s-", label="1F1B")
    ax.plot([r[0] for r in rows], [r[3] for r in rows], "k--", label="(p-1)/(m+p-1)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("microbatches")
    ax.set_ylabel("bubble fraction")
    ax.set_title(f"Pipeline bubble vs microbatches (p={stages})")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(Path(__file__).parent / "bubble.png", dpi=120)
    print("saved bubble.csv and bubble.png")


if __name__ == "__main__":
    main()
