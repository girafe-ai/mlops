"""Shared U-Net-like PyTorch model for the course segmentation example."""

import torch
from torch import nn
from torch.nn import functional


class TinyUNet(nn.Module):
    """Small encoder-decoder segmentation model with two skip connections.

    Args:
        num_classes: Number of contiguous class IDs predicted by the model.
        base_channels: Width of the first encoder block.
    """

    def __init__(self, num_classes: int, base_channels: int = 16) -> None:
        super().__init__()
        self.encoder_one = _ConvBlock(3, base_channels)
        self.encoder_two = _ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = _ConvBlock(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.up_two = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 2, kernel_size=2, stride=2
        )
        self.decoder_two = _ConvBlock(base_channels * 4, base_channels * 2)
        self.up_one = nn.ConvTranspose2d(
            base_channels * 2, base_channels, kernel_size=2, stride=2
        )
        self.decoder_one = _ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-class logits with the input spatial dimensions."""
        encoder_one = self.encoder_one(images)
        encoder_two = self.encoder_two(self.pool(encoder_one))
        bottleneck = self.bottleneck(self.pool(encoder_two))

        decoder_two = self.up_two(bottleneck)
        decoder_two = _resize_to(decoder_two, encoder_two)
        decoder_two = self.decoder_two(torch.cat((encoder_two, decoder_two), dim=1))
        decoder_one = self.up_one(decoder_two)
        decoder_one = _resize_to(decoder_one, encoder_one)
        decoder_one = self.decoder_one(torch.cat((encoder_one, decoder_one), dim=1))
        return self.head(decoder_one)


class _ConvBlock(nn.Sequential):
    """Two convolution/ReLU pairs used by the encoder and decoder."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


def _resize_to(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Match a decoder feature map to a skip connection after odd-size pooling."""
    if source.shape[-2:] == reference.shape[-2:]:
        return source
    return functional.interpolate(
        source, size=reference.shape[-2:], mode="bilinear", align_corners=False
    )
