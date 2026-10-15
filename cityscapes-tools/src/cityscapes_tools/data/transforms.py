"""Joint PyTorch image and mask transforms for semantic segmentation."""

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transforms_functional


class SegmentationTransform:
    """Resize an image/mask pair and optionally flip both with one random draw.

    Args:
        image_size: Output ``(height, width)`` for image and mask.
        random_horizontal_flip: Whether training augmentation may flip a pair.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        random_horizontal_flip: bool = False,
    ) -> None:
        self.image_size = image_size
        self.random_horizontal_flip = random_horizontal_flip

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a float image tensor and a long label-ID mask tensor."""
        image = transforms_functional.resize(
            image,
            size=self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = transforms_functional.resize(
            mask,
            size=self.image_size,
            interpolation=InterpolationMode.NEAREST,
        )
        if self.random_horizontal_flip and torch.rand(()) < 0.5:
            image = transforms_functional.hflip(image)
            mask = transforms_functional.hflip(mask)

        return (
            transforms_functional.pil_to_tensor(image).float() / 255,
            transforms_functional.pil_to_tensor(mask).squeeze(0).long(),
        )
