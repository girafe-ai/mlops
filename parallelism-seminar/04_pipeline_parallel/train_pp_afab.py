"""Pipeline parallel training with the AFAB (all-forward-all-backward) schedule.

We split ``TinyTransformer`` into two stages (half the blocks each) and
run it with ``torch.distributed.pipelining.ScheduleGPipe`` (PyTorch calls
this schedule "GPipe"). Only the first-stage rank feeds inputs; only the
last-stage rank sees the loss.

Launch:
    ./00_setup/launch.sh 2 04_pipeline_parallel/train_pp_afab.py
"""

import csv
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "00_setup"))
sys.path.insert(0, str(HERE.parent / "01_baseline"))

from conf_utils import load_config  # noqa: E402
from data import RandomTokenDataset  # noqa: E402
from dist_utils import cleanup_dist, is_main, print_rank0, setup_dist  # noqa: E402
from model import ModelConfig, TinyTransformer  # noqa: E402


class Stage0(nn.Module):
    """Embeddings + first half of the blocks."""

    def __init__(self, full: TinyTransformer, split_at: int):
        super().__init__()
        self.tok_emb = full.tok_emb
        self.pos_emb = full.pos_emb
        self.blocks = nn.ModuleList(list(full.blocks)[:split_at])

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        for block in self.blocks:
            x = block(x)
        return x


class Stage1(nn.Module):
    """Remaining blocks + final norm + lm head."""

    def __init__(self, full: TinyTransformer, split_at: int):
        super().__init__()
        self.blocks = nn.ModuleList(list(full.blocks)[split_at:])
        self.ln_f = full.ln_f
        self.head = full.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


def main():
    rank, world_size, local_rank = setup_dist()
    assert world_size == 2, "this script hard-codes a 2-stage split"
    cfg = load_config()
    device = torch.device("cuda", local_rank)
    torch.cuda.reset_peak_memory_stats(device)

    mcfg = ModelConfig.from_hydra(cfg)
    split_at = mcfg.n_layer // 2
    global_batch = cfg.train.batch_size
    num_microbatches = cfg.pipeline.num_microbatches
    micro_batch = global_batch // num_microbatches
    steps = cfg.train.steps

    full = TinyTransformer(mcfg)
    stage_mod = (Stage0(full, split_at) if rank == 0 else Stage1(full, split_at)).to(
        device
    )
    del full

    example_input = torch.zeros(
        micro_batch, mcfg.block_size, dtype=torch.long, device=device
    )
    if rank == 1:
        example_input = torch.zeros(
            micro_batch, mcfg.block_size, mcfg.d_model, device=device
        )

    stage = PipelineStage(
        stage_mod,
        stage_index=rank,
        num_stages=world_size,
        device=device,
        input_args=(example_input,),
    )

    def loss_fn(logits, targets):
        return nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
        )

    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=loss_fn)
    optim = torch.optim.AdamW(stage_mod.parameters(), lr=cfg.train.lr)

    dataset = RandomTokenDataset(
        num_samples=global_batch * steps,
        block_size=mcfg.block_size,
        vocab_size=mcfg.vocab_size,
        seed=cfg.train.seed,
    )
    loader = DataLoader(dataset, batch_size=global_batch, shuffle=False)

    print_rank0(f"pp afab  stages={world_size}  microbatches={num_microbatches}")

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    total_tokens = 0
    last_loss_t = torch.zeros((), device=device)

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        if rank == 0:
            schedule.step(x)
        else:
            losses: list[torch.Tensor] = []
            schedule.step(target=y, losses=losses)
            last_loss_t = torch.stack(losses).mean()
        optim.step()
        total_tokens += x.numel()
        if step % cfg.train.log_interval == 0 and rank == world_size - 1:
            print(f"step {step:3d}  loss {last_loss_t.item():.3f}")

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0
    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    tokens_per_sec = total_tokens / elapsed

    # Loss lives on the last stage — broadcast to rank 0 for logging.
    dist.broadcast(last_loss_t, src=world_size - 1)

    print_rank0(f"elapsed        : {elapsed:.2f} s")
    print_rank0(f"tokens/sec     : {tokens_per_sec:,.0f}")
    print_rank0(f"peak memory    : {peak_mem_gb:.2f} GiB  (rank {rank})")
    print_rank0(f"final loss     : {last_loss_t.item():.3f}")

    if is_main():
        out = HERE.parent / "metrics.csv"
        with out.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "pp_afab",
                    world_size,
                    f"{elapsed:.2f}",
                    f"{tokens_per_sec:.0f}",
                    f"{peak_mem_gb:.3f}",
                    f"{last_loss_t.item():.3f}",
                ]
            )

    cleanup_dist()


if __name__ == "__main__":
    main()
