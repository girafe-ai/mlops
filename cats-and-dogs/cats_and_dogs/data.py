from typing import Any

import torch
import torchvision
from torchvision import transforms
import lightning as L


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


def init_dataloader(dataset: Any, batch_size: int, shuffle: bool = True, num_workers: int = 6):
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


def init_predict_dataset(data_dir):
    raise NotImplementedError


class CatsAndDogsDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_batch_size=None,
        predict_batch_size=None,
        train_data_dir=None,
        val_data_dir=None,
        test_data_dir=None,
        predict_data_dir=None,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.predict_batch_size = predict_batch_size

        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.predict_data_dir = predict_data_dir

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = init_dataset(self.train_data_dir)
            self.val_dataset = init_dataset(self.val_data_dir)
        elif stage == "validate":
            self.val_dataset = init_dataset(self.val_data_dir)
        elif stage == "test":
            self.test_dataset = init_dataset(self.test_data_dir)
        elif stage == "predict":
            self.predict_dataset = init_predict_dataset(self.predict_data_dir)

    def train_dataloader(self):
        return init_dataloader(self.train_dataset, self.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return init_dataloader(self.val_dataset, self.predict_batch_size, shuffle=False)

    def test_dataloader(self):
        return init_dataloader(self.test_dataset, self.predict_batch_size, shuffle=False)

    def predict_dataloader(self):
        return init_dataloader(self.predict_dataset, self.predict_batch_size, shuffle=False)
