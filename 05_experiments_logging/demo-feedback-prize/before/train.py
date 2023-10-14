import torch
import hydra
from omegaconf import DictConfig
import transformers

from fbp.model import MyModel
from fbp.data import get_dataloders
from fbp.loops import train_loop


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = MyModel(
        model_name=cfg.model.name,
        num_target_cols=len(cfg.labels),
        dropout=cfg.model.dropout,
        freeze_backbone=cfg.model.freeze_backbone,
    ).to(device)

    train_dataloader, val_dataloader = get_dataloders(
        csv_path=cfg.data.csv_path,
        val_size=cfg.data.val_size,
        dataloader_num_wokers=cfg.data.dataloader_num_wokers,
        batch_size=cfg.data.batch_size,
        tokenizer_model_name=cfg.model.name,
        text_max_length=cfg.data.text_max_length,
        labels=cfg.labels
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
    )
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.scheduler.num_warmup_steps,
        num_training_steps=cfg.scheduler.num_training_steps,
    )

    loss_fn = torch.nn.SmoothL1Loss()

    train_loop(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=cfg.train.num_epochs,
        use_amp=cfg.train.use_amp,
        grad_accum_steps=cfg.train.grad_accum_steps,
        loss_fn=loss_fn,
    )


if __name__ == "__main__":
    main()
