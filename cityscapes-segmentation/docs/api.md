# API Reference

## raw/

### Data

::: cityscapes-segmentation.raw.data
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
      members:
        - CityscapesDataset
        - get_dataloader

### Training

::: cityscapes-segmentation.raw.train
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
      members:
        - train
        - train_one_epoch
        - validate

### Inference

::: cityscapes-segmentation.raw.infer
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
      members:
        - run

### Utilities

::: cityscapes-segmentation.raw.utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
      members:
        - build_transforms
        - compute_miou

## lightning/

### Model

::: cityscapes-segmentation.lightning.model
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
      members:
        - SegmentationModel

### Data

::: cityscapes-segmentation.lightning.data
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
      members:
        - CityscapesDataset
        - CityscapesDataModule

### Training

::: cityscapes-segmentation.lightning.train
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
      members:
        - train

### Inference

::: cityscapes-segmentation.lightning.infer
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
      members:
        - run

### Utilities

::: cityscapes-segmentation.lightning.utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 4
      members:
        - build_transforms
        - compute_miou
