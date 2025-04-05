from typing import Any

import torch
import torchvision
import torchvision.transforms as transforms


def init_dataset(path: str):
    """Initialize torch dataset from folder

    Args:
        path (str): path to the folder with images

    Returns:
        torchvision.datasets.ImageFolder: usable torch dataset
    """
    transformer = transforms.Compose(
        [
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return torchvision.datasets.ImageFolder(path, transformer)


def init_dataloader(
    dataset: Any, batch_size: int, shuffle: bool = True, num_workers: int = 6
):
    """Initialize torch dataloader from dataset

    Args:
        dataset (Any): dataset for dataloader
        batch_size (int): -
        shuffle (bool, optional): flag for shuffling data. Defaults to True.
        num_workers (int, optional): Defaults to 6.

    Returns:
        torch.utils.data.Dataloader: usable torch dataloader
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
