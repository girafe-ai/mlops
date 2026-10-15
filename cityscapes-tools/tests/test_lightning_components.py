import csv
import json

import torch
from cityscapes_tools.training.lightning.datamodule import CityscapesDataModule
from cityscapes_tools.training.lightning.module import CityscapesLightningModule
from PIL import Image


def test_lightning_module_configures_adamw_optimizer(data_config) -> None:
    module = CityscapesLightningModule(
        num_classes=3,
        base_channels=4,
        learning_rate=0.01,
        ignore_index=data_config.ignore_index,
    )

    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert module.loss_function.ignore_index == data_config.ignore_index
    for metric in (module.train_iou, module.validation_iou, module.test_iou):
        assert metric.ignore_index == data_config.ignore_index


def test_datamodule_creates_train_and_validation_batches(tmp_path, data_config) -> None:
    manifest_path = tmp_path / "filtered_manifest.csv"
    rows = []
    for split in ("train", "val"):
        image_path = (
            tmp_path / "leftImg8bit" / split / "aachen" / f"{split}_leftImg8bit.png"
        )
        annotation_path = (
            tmp_path / "gtFine" / split / "aachen" / f"{split}_gtFine_polygons.json"
        )
        label_path = annotation_path.with_name(f"{split}_gtFine_labelIds.png")
        image_path.parent.mkdir(parents=True)
        annotation_path.parent.mkdir(parents=True)
        Image.new("RGB", (4, 3), color=(10, 20, 30)).save(image_path)
        Image.fromarray(
            torch.tensor([[7, 24, 26, 33]] * 3, dtype=torch.uint8).numpy()
        ).save(label_path)
        annotation_path.write_text(json.dumps({"objects": []}), encoding="utf-8")
        rows.append(
            {
                "image_id": f"{split}/aachen/{split}",
                "split": split,
                "city": "aachen",
                "image_path": image_path,
                "annotation_path": annotation_path,
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    data_module = CityscapesDataModule(
        manifest_path=manifest_path,
        class_ids={7: 0, 24: 1, 26: 2},
        image_size=(6, 8),
        batch_size=1,
        num_workers=0,
        ignore_index=data_config.ignore_index,
    )
    data_module.prepare_data()
    data_module.setup("fit")

    train_loader = data_module.train_dataloader()
    train_batch = next(iter(train_loader))
    validation_batch = next(iter(data_module.val_dataloader()))

    assert train_batch["images"].shape == (1, 3, 6, 8)
    assert validation_batch["masks"].shape == (1, 6, 8)
    assert train_loader.pin_memory is torch.cuda.is_available()
    assert data_module.train_dataset.ignore_index == data_config.ignore_index
    assert data_module.val_dataset.ignore_index == data_config.ignore_index
    assert data_config.ignore_index in train_batch["masks"].unique().tolist()
    assert train_loader.collate_fn.keywords["ignore_index"] == data_config.ignore_index
    for stage, loader_method in (
        ("validate", data_module.val_dataloader),
        ("test", data_module.test_dataloader),
        ("predict", data_module.predict_dataloader),
    ):
        data_module.setup(stage)
        loader = loader_method()
        assert loader.dataset.ignore_index == data_config.ignore_index
        assert loader.collate_fn.keywords["ignore_index"] == data_config.ignore_index
        assert data_config.ignore_index in next(iter(loader))["masks"].unique().tolist()
