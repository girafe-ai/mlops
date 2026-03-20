# API Reference

## Data

::: cityscapes-segmentation.data
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - CityscapesDataset
        - get_dataloader

## Training

::: cityscapes-segmentation.train
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - train
        - train_one_epoch
        - validate

## Inference

::: cityscapes-segmentation.infer
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - run

## Utilities

::: cityscapes-segmentation.utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      heading_level: 3
      members:
        - build_transforms
        - compute_miou
