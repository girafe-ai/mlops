"""Shared distributed helpers used by every part of the seminar."""

import os

import torch
import torch.distributed as dist


def setup_dist(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize the default process group from torchrun env vars.

    Returns (rank, world_size, local_rank).
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return rank, world_size, local_rank


def cleanup_dist() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def print_rank0(*args, **kwargs) -> None:
    if is_main():
        print(*args, **kwargs)
