"""TinyTransformer with a hand-written tensor-parallel MLP.

Only the MLP block is parallelized — attention stays replicated. This is
intentional: it keeps the diff small and isolates the column->row pairing
that the mini-lecture introduced.

Launch:
    ./00_setup/launch.sh 2 03_tensor_parallel/train_tp_manual.py
"""

import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "00_setup"))
sys.path.insert(0, str(HERE.parent / "01_baseline"))

from conf_utils import load_config  # noqa: E402
from data import RandomTokenDataset  # noqa: E402
from dist_utils import cleanup_dist, print_rank0, setup_dist  # noqa: E402
from model import CausalSelfAttention, ModelConfig, TinyTransformer  # noqa: E402
from tp_layers import ColumnParallelLinear, RowParallelLinear  # noqa: E402


class TPBlock(nn.Module):
    """Transformer block where the MLP is column->row parallelized."""

    def __init__(self, mcfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(mcfg.d_model)
        self.attn = CausalSelfAttention(mcfg)
        self.ln2 = nn.LayerNorm(mcfg.d_model)
        self.fc1 = ColumnParallelLinear(mcfg.d_model, mcfg.d_ff, gather_output=False)
        self.fc2 = RowParallelLinear(mcfg.d_ff, mcfg.d_model, input_is_parallel=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        h = F.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


def build_tp_model(mcfg: ModelConfig) -> TinyTransformer:
    model = TinyTransformer(mcfg)
    model.blocks = nn.ModuleList([TPBlock(mcfg) for _ in range(mcfg.n_layer)])
    return model


def main():
    _, world_size, local_rank = setup_dist()
    cfg = load_config()
    device = torch.device("cuda", local_rank)
    torch.cuda.reset_peak_memory_stats(device)

    mcfg = ModelConfig.from_hydra(cfg)
    batch_size = cfg.train.batch_size
    steps = cfg.train.steps

    model = build_tp_model(mcfg).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)

    dataset = RandomTokenDataset(
        num_samples=batch_size * steps,
        block_size=mcfg.block_size,
        vocab_size=mcfg.vocab_size,
        seed=cfg.train.seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print_rank0(f"tp world_size: {world_size}  batch: {batch_size}")

    model.train()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    total_tokens = 0
    last_loss = 0.0

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        last_loss = loss.item()
        total_tokens += x.numel()
        if step % cfg.train.log_interval == 0:
            print_rank0(f"step {step:3d}  loss {last_loss:.3f}")

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3

    tokens_per_sec = total_tokens / elapsed
    print_rank0(f"elapsed        : {elapsed:.2f} s")
    print_rank0(f"tokens/sec     : {tokens_per_sec:,.0f}  (single replica)")
    print_rank0(f"peak memory    : {peak_mem_gb:.2f} GiB")
    print_rank0(f"final loss     : {last_loss:.3f}")

    from dist_utils import is_main  # noqa: E402

    if is_main():
        out = HERE.parent / "metrics.csv"
        with out.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "tp_manual",
                    world_size,
                    f"{elapsed:.2f}",
                    f"{tokens_per_sec:.0f}",
                    f"{peak_mem_gb:.3f}",
                    f"{last_loss:.3f}",
                ]
            )

    cleanup_dist()


if __name__ == "__main__":
    main()
