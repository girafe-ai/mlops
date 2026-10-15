# Cityscapes tools

This project downloads Cityscapes, prepares DVC metadata through PostgreSQL,
and demonstrates semantic segmentation with raw PyTorch and Lightning.

Read the [dataset overview](dataset.md) for details about Cityscapes, then follow
the [download guide](download.md) to set up an account and get the data.
Continue with the [data pipeline](pipeline.md) and the [training walkthrough](training.md).
Console entry points live in `cityscapes_tools.cli`; shared data/model components
and the two training implementations are separate Python subpackages.
