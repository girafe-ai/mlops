"""Same MLP tensor-parallelism, but via torch.distributed.tensor.parallel.

After implementing the hand-written column/row layers, show that the
built-in ``parallelize_module`` API does exactly the same thing with a
plan dictionary — no custom Module subclasses required.

Launch:
    ./00_setup/launch.sh 2 03_tensor_parallel/train_tp_dtensor.py
"""

import csv
import sys
import time
from pathlib import Path

import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "00_setup"))
sys.path.insert(0, str(HERE.parent / "01_baseline"))

from conf_utils import load_config  # noqa: E402
from data import RandomTokenDataset  # noqa: E402
from dist_utils import cleanup_dist, print_rank0, setup_dist  # noqa: E402
from model import ModelConfig, TinyTransformer  # noqa: E402


def main():
    _, world_size, local_rank = setup_dist()
    cfg = load_config()
    device = torch.device("cuda", local_rank)
    torch.cuda.reset_peak_memory_stats(device)

    mcfg = ModelConfig.from_hydra(cfg)
    batch_size = cfg.train.batch_size
    steps = cfg.train.steps

    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("tp",))

    model = TinyTransformer(mcfg).to(device)

    for block in model.blocks:
        parallelize_module(
            block,
            mesh,
            {
                "mlp.fc1": ColwiseParallel(),
                "mlp.fc2": RowwiseParallel(),
            },
        )

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, foreach=False)
    dataset = RandomTokenDataset(
        num_samples=batch_size * steps,
        block_size=mcfg.block_size,
        vocab_size=mcfg.vocab_size,
        seed=cfg.train.seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print_rank0(f"dtensor tp world_size: {world_size}")

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
    print_rank0(f"tokens/sec     : {tokens_per_sec:,.0f}")
    print_rank0(f"peak memory    : {peak_mem_gb:.2f} GiB")
    print_rank0(f"final loss     : {last_loss:.3f}")

    from dist_utils import is_main  # noqa: E402

    if is_main():
        out = HERE.parent / "metrics.csv"
        with out.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "tp_dtensor",
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
