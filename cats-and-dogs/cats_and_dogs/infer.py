import torch
import lightning as L

from cats_and_dogs.data import CatsAndDogsDataModule
from cats_and_dogs.model import SimpleClassifier
from cats_and_dogs.module import CatsAndDogsModule


def infer(model_file: str, data_dir: str, batch_size: int = 32):
    datamodule = CatsAndDogsDataModule(predict_batch_size=batch_size, test_data_dir=data_dir)
    module = CatsAndDogsModule(SimpleClassifier())

    module.model.load_state_dict(torch.load(model_file, weights_only=True))

    trainer = L.Trainer()
    trainer.test(module, datamodule=datamodule)
