import torch
from cityscapes_tools.models.tiny_unet import TinyUNet


def test_tiny_unet_returns_logits_for_every_input_pixel() -> None:
    model = TinyUNet(num_classes=3, base_channels=4)

    logits = model(torch.rand((2, 3, 32, 48)))

    assert logits.shape == (2, 3, 32, 48)
