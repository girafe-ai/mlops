"""Single-GPU baseline training loop.

Establishes the reference numbers (tokens/sec, peak memory, step time)
that every parallel variant in later parts will be compared against.

Run:
    python train.py
    python train.py train.batch_size=32 model.n_layer=12
"""

import csv
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "00_setup"))

from conf_utils import load_config  # noqa: E402
from data import RandomTokenDataset  # noqa: E402
from model import ModelConfig, TinyTransformer  # noqa: E402


def main():
    cfg = load_config()

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    torch.cuda.reset_peak_memory_stats(device)

    mcfg = ModelConfig.from_hydra(cfg)
    batch_size = cfg.train.batch_size
    steps = cfg.train.steps
    lr = cfg.train.lr

    model = TinyTransformer(mcfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset = RandomTokenDataset(
        num_samples=batch_size * steps,
        block_size=mcfg.block_size,
        vocab_size=mcfg.vocab_size,
        seed=cfg.train.seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"params: {model.num_params() / 1e6:.1f}M")

    model.train()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    total_tokens = 0
    losses = []

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        total_tokens += x.numel()
        if step % cfg.train.log_interval == 0:
            print(f"step {step:3d}  loss {loss.item():.3f}")

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    tokens_per_sec = total_tokens / elapsed

    print()
    print(f"elapsed        : {elapsed:.2f} s")
    print(f"tokens/sec     : {tokens_per_sec:,.0f}")
    print(f"peak memory    : {peak_mem_gb:.2f} GiB")
    print(f"final loss     : {losses[-1]:.3f}")

    out = HERE.parent / "metrics.csv"
    with out.open("a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "baseline",
                1,
                f"{elapsed:.2f}",
                f"{tokens_per_sec:.0f}",
                f"{peak_mem_gb:.3f}",
                f"{losses[-1]:.3f}",
            ]
        )


if __name__ == "__main__":
    main()
