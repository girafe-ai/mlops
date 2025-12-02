import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
import hydra
import os
from omegaconf import DictConfig, OmegaConf

from cats_and_dogs.data import CatsAndDogsDataModule
from cats_and_dogs.model import SimpleClassifier, ConvClassifier
from cats_and_dogs.module import CatsAndDogsModule


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    print(cfg)
    datamodule = CatsAndDogsDataModule(
        cfg.data.train_batch_size,
        cfg.data.predict_batch_size,
        cfg.data.train_data_dir,
        cfg.data.val_data_dir,
    )
    if cfg.model.type == "conv":
        model = ConvClassifier(cfg.model.num_classes)
    elif cfg.model.type == "simple":
        model = SimpleClassifier()

    module = CatsAndDogsModule(model)

    logger = TensorBoardLogger("tb_logs", name=cfg.logging.model_name)
    trainer = L.Trainer(max_epochs=cfg.num_epochs, logger=logger)
    trainer.fit(module, datamodule=datamodule)
    torch.save(module.model.state_dict(), cfg.output_file)


if __name__ == "__main__":
    train()
