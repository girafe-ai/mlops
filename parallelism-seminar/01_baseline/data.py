"""Synthetic token dataset.

We deliberately avoid downloading a real corpus — the seminar is about
parallelism, not tokenization. A deterministic random stream of token IDs
keeps every part reproducible and removes network dependencies.
"""

import torch
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset):
    def __init__(
        self, num_samples: int, block_size: int, vocab_size: int, seed: int = 0
    ):
        self.num_samples = num_samples
        self.block_size = block_size
        self.vocab_size = vocab_size
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(
            0, vocab_size, (num_samples, block_size + 1), generator=g
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        row = self.data[idx]
        return row[:-1], row[1:]
