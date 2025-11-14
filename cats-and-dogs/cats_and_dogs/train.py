import torch
import lightning as L

from cats_and_dogs.data import CatsAndDogsDataModule
from cats_and_dogs.model import SimpleClassifier
from cats_and_dogs.module import CatsAndDogsModule


def train(
    train_data_dir: str,
    val_data_dir: str,
    output_file: str,
    batch_size: int = 32,
    num_epochs: int = 1,
):
    datamodule = CatsAndDogsDataModule(batch_size, batch_size, train_data_dir, val_data_dir)
    module = CatsAndDogsModule(SimpleClassifier())

    trainer = L.Trainer(max_epochs=num_epochs)
    trainer.fit(module, datamodule=datamodule)
    torch.save(module.model.state_dict(), output_file)
