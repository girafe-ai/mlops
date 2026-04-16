"""FullyShardedDataParallel version of the DDP script.

Key difference vs train_ddp.py: parameters, gradients, and optimizer states
are sharded across ranks instead of replicated. Peak per-GPU memory should
drop noticeably compared to DDP on the same model.

Launch:
    ./00_setup/launch.sh 2 02_data_parallel/train_fsdp.py
"""

import csv
import functools
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "00_setup"))
sys.path.insert(0, str(HERE.parent / "01_baseline"))

from conf_utils import load_config  # noqa: E402
from data import RandomTokenDataset  # noqa: E402
from dist_utils import cleanup_dist, is_main, print_rank0, setup_dist  # noqa: E402
from model import Block, ModelConfig, TinyTransformer  # noqa: E402


def main():
    rank, world_size, local_rank = setup_dist()
    cfg = load_config()
    device = torch.device("cuda", local_rank)
    torch.cuda.reset_peak_memory_stats(device)

    mcfg = ModelConfig.from_hydra(cfg)
    per_rank_batch = cfg.train.batch_size
    global_batch = per_rank_batch * world_size
    steps = cfg.train.steps
    lr = cfg.train.lr

    model = TinyTransformer(mcfg).to(device)

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block},
    )
    model = FSDP(model, auto_wrap_policy=wrap_policy, device_id=local_rank)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset = RandomTokenDataset(
        num_samples=global_batch * steps,
        block_size=mcfg.block_size,
        vocab_size=mcfg.vocab_size,
        seed=cfg.train.seed,
    )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    loader = DataLoader(dataset, batch_size=per_rank_batch, sampler=sampler)

    print_rank0(f"world_size: {world_size}  per-rank batch: {per_rank_batch}")
    print_rank0(model)

    model.train()
    sampler.set_epoch(0)
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    total_tokens_local = 0
    last_loss = 0.0

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        last_loss = loss.item()
        total_tokens_local += x.numel()
        if step % cfg.train.log_interval == 0:
            print_rank0(f"step {step:3d}  loss {last_loss:.3f}")

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    tokens_tensor = torch.tensor(
        [total_tokens_local], device=device, dtype=torch.float64
    )
    dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
    total_tokens_global = tokens_tensor.item()

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    tokens_per_sec = total_tokens_global / elapsed

    if is_main():
        print()
        print(f"elapsed        : {elapsed:.2f} s")
        print(f"tokens/sec     : {tokens_per_sec:,.0f}  (global)")
        print(f"peak memory    : {peak_mem_gb:.2f} GiB  (rank 0)")
        print(f"final loss     : {last_loss:.3f}")

        out = HERE.parent / "metrics.csv"
        with out.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "fsdp",
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
