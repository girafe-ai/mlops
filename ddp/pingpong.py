import os
import random

import torch.distributed as dist
from torch.multiprocessing import Process


def init_process(rank, size, fn, port, backend="gloo"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


def run_pingpong(rank, size, num_iter=10):
    for i in range(num_iter):
        # TODO: write ping-pong logic
        if rank == 0:
            # Процесс 0: отправляет "Ping"
            print(f"Iter {i}: Rank 0 -> Ping")
            dist.barrier()  # Ждём, пока Rank 1 получит "Ping"
            dist.barrier()  # Ждём, пока Rank 1 отправит "Pong"
        else:
            # Процесс 1: ждёт "Ping", отправляет "Pong"
            dist.barrier()  # Ждём, пока Rank 0 отправит "Ping"
            print(f"Iter {i}: Rank 1 -> Pong")
            dist.barrier()  # Ждём, пока Rank 0 получит "Pong"


if __name__ == "__main__":
    size = 2
    processes = []
    port = random.randint(8888, 9000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_pingpong, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
