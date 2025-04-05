import pytest
import torch
import torchvision

from cats_and_dogs.pl_modules.data import MyDataModule


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        self.dataset = torchvision.datasets.ImageFolder(path, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


@pytest.fixture
def datamodule(tmp_path):
    test_config = {
        "data_loading": {
            "train_data_path": "data/train_11k",
            "val_data_path": "data/val",
            "test_data_path": "data/test_labeled",
        },
        "training": {"batch_size": 2, "num_workers": 1},
        "model": {
            "image_height": 32,
            "image_width": 32,
            "image_mean": [0.0, 0.0, 0.0],
            "image_std": [0.1, 0.1, 0.1],
        },
    }

    dm = MyDataModule(config=test_config)
    dm.setup()
    return dm


@pytest.mark.requires_files
def test_dataloaders(datamodule):
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    images, labels = batch
    assert images.shape[0] == 2, "First shape should be batch_size!"
    assert len(labels) == 2, "There should be only 2 labels!"
    assert set(labels.numpy()) <= {0, 1}, "Class should be 0 or 1!"
