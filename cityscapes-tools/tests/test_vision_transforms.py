import torch
from cityscapes_tools.data.transforms import SegmentationTransform
from PIL import Image


def test_segmentation_transform_resizes_mask_with_nearest_neighbor() -> None:
    image = Image.new("RGB", (3, 2), color=(10, 20, 30))
    mask = Image.fromarray(
        torch.tensor([[7, 24, 26], [33, 7, 24]], dtype=torch.uint8).numpy()
    )
    transform = SegmentationTransform(image_size=(4, 6), random_horizontal_flip=False)

    transformed_image, transformed_mask = transform(image, mask)

    assert transformed_image.shape == (3, 4, 6)
    assert transformed_mask.shape == (4, 6)
    assert set(transformed_mask.unique().tolist()) == {7, 24, 26, 33}
