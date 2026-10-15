"""Padding collation for variable-sized semantic segmentation samples."""

import torch


def pad_collate(
    batch: list[dict[str, object]],
    ignore_index: int,
) -> dict[str, object]:
    """Pad images with zeros and masks with the configured loss/metric ignore index."""
    max_height = max(sample["image"].shape[-2] for sample in batch)
    max_width = max(sample["image"].shape[-1] for sample in batch)
    images = []
    masks = []
    image_ids = []

    for sample in batch:
        image = sample["image"]
        mask = sample["mask"]
        image_padding = (
            0,
            max_width - image.shape[-1],
            0,
            max_height - image.shape[-2],
        )
        mask_padding = (0, max_width - mask.shape[-1], 0, max_height - mask.shape[-2])
        images.append(torch.nn.functional.pad(image, image_padding, value=0))
        masks.append(torch.nn.functional.pad(mask, mask_padding, value=ignore_index))
        image_ids.append(sample["image_id"])

    return {
        "images": torch.stack(images),
        "masks": torch.stack(masks),
        "image_ids": image_ids,
    }
