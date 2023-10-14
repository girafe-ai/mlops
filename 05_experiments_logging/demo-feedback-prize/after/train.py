import os

import hydra
import torch
from omegaconf import DictConfig
import lightning.pytorch as pl

from fbp.model import MyModel
from fbp.data import MyDataModule


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")
    dm = MyDataModule(
        csv_path=cfg.data.csv_path,
        val_size=cfg.data.val_size,
        dataloader_num_wokers=cfg.data.dataloader_num_wokers,
        batch_size=cfg.data.batch_size,
        tokenizer_model_name=cfg.model.name,
        text_max_length=cfg.data.text_max_length,
        labels=cfg.labels,
    )
    model = MyModel(cfg)

    loggers = [
        pl.loggers.CSVLogger("./.logs/my-csv-logs", name=cfg.artifacts.experiment_name),
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri="file:./.logs/my-mlflow-logs",
        ),
        pl.loggers.TensorBoardLogger(
            "./.logs/my-tb-logs", name=cfg.artifacts.experiment_name
        ),
        pl.loggers.WandbLogger(
            project="mlops-logging-demo", name=cfg.artifacts.experiment_name
        ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=cfg.callbacks.model_summary.max_depth),
    ]

    if cfg.callbacks.swa.use:
        callbacks.append(
            pl.callbacks.StochasticWeightAveraging(swa_lrs=cfg.callbacks.swa.lrs)
        )

    if cfg.artifacts.checkpoint.use:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(
                    cfg.artifacts.checkpoint.dirpath, cfg.artifacts.experiment_name
                ),
                filename=cfg.artifacts.checkpoint.filename,
                monitor=cfg.artifacts.checkpoint.monitor,
                save_top_k=cfg.artifacts.checkpoint.save_top_k,
                every_n_train_steps=cfg.artifacts.checkpoint.every_n_train_steps,
                every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
            )
        )

    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        max_steps=cfg.train.num_warmup_steps + cfg.train.num_training_steps,
        accumulate_grad_batches=cfg.train.grad_accum_steps,
        val_check_interval=cfg.train.val_check_interval,
        overfit_batches=cfg.train.overfit_batches,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        deterministic=cfg.train.full_deterministic_mode,
        benchmark=cfg.train.benchmark,
        gradient_clip_val=cfg.train.gradient_clip_val,
        profiler=cfg.train.profiler,
        log_every_n_steps=cfg.train.log_every_n_steps,
        detect_anomaly=cfg.train.detect_anomaly,
        enable_checkpointing=cfg.artifacts.checkpoint.use,
        logger=loggers,
        callbacks=callbacks,
    )

    if cfg.train.batch_size_finder:
        tuner = pl.tuner.Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode="power")

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
