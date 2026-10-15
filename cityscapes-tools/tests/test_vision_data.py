import csv
import json

import pytest
import torch
from cityscapes_tools.data.cityscapes import remap_cityscapes_mask
from cityscapes_tools.data.collate import pad_collate
from cityscapes_tools.data.loaders import create_dataloader, create_dataset
from PIL import Image


def test_remap_cityscapes_mask_keeps_only_configured_classes(data_config) -> None:
    mask = torch.tensor([[7, 24], [26, 33]], dtype=torch.long)

    remapped = remap_cityscapes_mask(
        mask, dict(data_config.class_ids), ignore_index=data_config.ignore_index
    )

    assert torch.equal(
        remapped,
        torch.tensor([[0, 1], [2, data_config.ignore_index]], dtype=torch.long),
    )


@pytest.fixture
def variable_size_samples():
    return [
        {
            "image": torch.ones((3, 2, 3)),
            "mask": torch.zeros((2, 3), dtype=torch.long),
            "image_id": "train/aachen/first",
        },
        {
            "image": torch.ones((3, 3, 2)),
            "mask": torch.ones((3, 2), dtype=torch.long),
            "image_id": "train/aachen/second",
        },
    ]


def test_pad_collate_pads_masks_with_ignore_index(
    variable_size_samples, data_config
) -> None:
    collated = pad_collate(variable_size_samples, ignore_index=data_config.ignore_index)

    assert collated["images"].shape == (2, 3, 3, 3)
    assert collated["masks"].shape == (2, 3, 3)
    assert collated["masks"][0, 2, 2].item() == data_config.ignore_index
    assert collated["masks"][1, 0, 2].item() == data_config.ignore_index
    assert collated["image_ids"] == ["train/aachen/first", "train/aachen/second"]


def test_dataloader_uses_configured_padding(variable_size_samples, data_config) -> None:
    loader = create_dataloader(
        variable_size_samples,
        batch_size=2,
        num_workers=0,
        ignore_index=data_config.ignore_index,
    )

    batch = next(iter(loader))

    assert batch["masks"][0, 2, 2].item() == data_config.ignore_index
    assert batch["masks"][1, 0, 2].item() == data_config.ignore_index


def test_dataset_loads_image_and_label_ids_from_filtered_manifest(
    tmp_path, data_config
) -> None:
    image_path = (
        tmp_path / "leftImg8bit" / "train" / "aachen" / "sample_leftImg8bit.png"
    )
    annotation_path = (
        tmp_path / "gtFine" / "train" / "aachen" / "sample_gtFine_polygons.json"
    )
    label_path = annotation_path.with_name("sample_gtFine_labelIds.png")
    image_path.parent.mkdir(parents=True)
    annotation_path.parent.mkdir(parents=True)
    Image.new("RGB", (3, 2), color=(10, 20, 30)).save(image_path)
    Image.fromarray(
        torch.tensor([[7, 24, 26], [33, 7, 24]], dtype=torch.uint8).numpy()
    ).save(label_path)
    annotation_path.write_text(json.dumps({"objects": []}), encoding="utf-8")

    manifest_path = tmp_path / "filtered_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=["image_id", "split", "city", "image_path", "annotation_path"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_id": "train/aachen/sample",
                "split": "train",
                "city": "aachen",
                "image_path": image_path,
                "annotation_path": annotation_path,
            }
        )

    dataset = create_dataset(
        manifest_path=manifest_path,
        split="train",
        class_ids={7: 0, 24: 1, 26: 2},
        image_size=(2, 3),
        ignore_index=data_config.ignore_index,
    )

    sample = dataset[0]

    assert sample["image"].shape == (3, 2, 3)
    assert sample["mask"].tolist() == [[0, 1, 2], [data_config.ignore_index, 0, 1]]
    assert sample["image_id"] == "train/aachen/sample"
