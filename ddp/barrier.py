import os

import torch
import torch.distributed as dist


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def worker(rank, world_size):
    setup(rank, world_size)

    print(f"Rank {rank} before barrier")
    dist.barrier()  # Все процессы ждут здесь
    print(f"Rank {rank} after barrier")

    cleanup()


def run():
    world_size = 2
    torch.multiprocessing.spawn(
        worker, args=(world_size,), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    run()
