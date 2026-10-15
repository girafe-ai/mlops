# PyTorch and Lightning

This lesson trains the same small semantic-segmentation model in two ways:
first with an explicit PyTorch loop and then with Lightning orchestration.
The data and model are shared; only the training orchestration changes.

## Package structure

```text
src/cityscapes_tools/
├── cli/                          # Thin console entry points
│   ├── download.py
│   ├── build_metadata.py
│   ├── materialize_dataset.py
│   ├── train_torch.py
│   └── train_lightning.py
├── data/                         # Shared, PyTorch-only data layer
│   ├── cityscapes.py              # Dataset and mask remapping
│   ├── transforms.py             # Paired image/mask transforms
│   ├── collate.py                # Padding collator
│   └── loaders.py                # Dataset/DataLoader factories
├── models/
│   └── tiny_unet.py               # Shared torch.nn.Module
├── training/
│   ├── torch/
│   │   ├── loop.py               # train_one_epoch, validate, device selection
│   │   └── runner.py             # Assemble data/model and run epochs
│   └── lightning/
│       ├── module.py             # LightningModule
│       ├── datamodule.py         # LightningDataModule
│       └── runner.py             # Trainer, logger, and callbacks
├── pipelines/                    # Download, metadata, PostgreSQL, and SQL
└── conf/                         # Packaged Hydra defaults and config groups
```

`cli/` reads command-line options and delegates work to the package. Training
logic lives in `training/`, so it can also be imported from Python. The raw
PyTorch path uses `data.loaders` directly and does not import Lightning.
Lightning's DataModule wraps those same factories.

## Two-hour demonstration

| Minutes | Topic and code to show                                                   |
| ------- | ------------------------------------------------------------------------ |
| 0–8     | DVC manifest, segmentation targets, shared components                    |
| 8–23    | `data/cityscapes.py`: Dataset, RGB/mask loading, label remapping         |
| 23–38   | `data/transforms.py` and `data/collate.py`: joint transforms and padding |
| 38–48   | `data/loaders.py`: batching, workers, shuffle, tensor shapes             |
| 48–58   | `models/tiny_unet.py`: logits and `CrossEntropyLoss`                     |
| 58–78   | `training/torch/`: explicit train/validation loop and device handling    |
| 78–93   | `training/lightning/module.py`: steps, optimizer, logging, IoU           |
| 93–105  | `training/lightning/datamodule.py`: data lifecycle and loaders           |
| 105–116 | `training/lightning/runner.py`: Trainer, checkpoint, early stopping      |
| 116–120 | Compare both paths and inspect a configuration override                  |

## Prepare the data

The training commands read the DVC-generated manifest at
`data/processed/filtered_manifest.csv`. It contains paths to the Cityscapes
images and fine annotation files.

```bash
uv run dvc pull  # when a DVC remote has been configured
uv run dvc repro
```

The example configuration keeps three original Cityscapes label IDs:

| Class  | Original label ID | Training ID |
| ------ | ----------------: | ----------: |
| road   |                 7 |           0 |
| person |                24 |           1 |
| car    |                26 |           2 |

All other original labels become `data.ignore_index` (default: `255`). This
value is not an additional class: those pixels are excluded from loss and IoU.
The same configured value is used for mask padding. Change the label mapping
and ignore index in `src/cityscapes_tools/conf/data/vision.yaml`; keep the ignore
index outside the model's class range (`0` through `num_classes - 1`).

Both runners pass this value explicitly to datasets, collators, loss, and
(in Lightning) segmentation metrics. Components receive an integer argument;
they do not read Hydra directly or fall back to a Python constant.

Inspect an alternative value without starting training:

```bash
uv run cityscapes-train-lightning --cfg job data.ignore_index=-1
```

## The PyTorch data layer

`CityscapesSegmentationDataset` reads an image and its `labelIds` mask from
one manifest row. `SegmentationTransform` resizes image and mask together;
the RGB image uses bilinear interpolation while the class mask uses nearest
neighbour interpolation. Applying an image-only random crop or flip to a
segmentation mask would misalign targets and images.

`pad_collate` demonstrates batching samples whose spatial dimensions differ:
it zero-pads images and pads masks with the configured ignore index, so padded pixels do not affect
`CrossEntropyLoss`.

## Explicit PyTorch loop

Run the manual implementation with:

```bash
uv run cityscapes-train-torch
```

The `training/torch/loop.py` module shows the complete loop: `model.train()`, moving a batch to a
device, forward pass, loss, `zero_grad()`, `backward()`, `optimizer.step()`,
then `model.eval()` and `torch.inference_mode()` for validation. Its settings
are in `src/cityscapes_tools/conf/torch.yaml` and
`src/cityscapes_tools/conf/training/manual.yaml`.

## Lightning implementation

The default Lightning logger is MLflow; start the server described in
[Local MLflow logging](#local-mlflow-logging) before training. For the CSV
demonstration, run:

```bash
uv run cityscapes-train-lightning logger=csv
```

`CityscapesLightningModule` owns the model and optimization logic:

- `forward` returns logits;
- `training_step`, `validation_step`, and `test_step` calculate loss and IoU;
- `predict_step` returns masks and source image IDs;
- `configure_optimizers` creates AdamW.

`CityscapesDataModule` owns data lifecycle and loaders:

- `prepare_data` verifies the DVC manifest rather than downloading data;
- `setup` creates datasets for a requested stage;
- `train_dataloader`, `val_dataloader`, `test_dataloader`, and
  `predict_dataloader` create loaders.

`Trainer` owns the outer loop. The Lightning runner configures automatic device
selection, a configured logger, `ModelCheckpoint`, and `EarlyStopping`. Its settings are in
`src/cityscapes_tools/conf/lightning.yaml` and
`src/cityscapes_tools/conf/training/lightning.yaml`.

The demonstration uses `val` for `test_dataloader` and `predict_dataloader` by
default, since official Cityscapes test targets are not available for evaluation.
Calling `trainer.test` here reuses validation data and is an API demonstration,
not an independent held-out score. The runner calls `fit` and `test`; the
DataModule and LightningModule also expose hooks for `validate` and `predict`.

### CSV logging

The CSV logger is defined in `src/cityscapes_tools/conf/logger/csv.yaml`:

```yaml
_target_: lightning.pytorch.loggers.CSVLogger
save_dir: lightning_logs
name: ${experiment.name}
version: ${experiment.run_id}
flush_logs_every_n_steps: 1
```

Hydra's `instantiate(config.logger)` builds the logger passed to `Trainer`.
Select a logger with `logger=csv` or `logger=mlflow` without changing the runner.

Experiment identity is configured in
`src/cityscapes_tools/conf/experiment/experiment.yaml`: `name` groups related
runs, while `run_id` defaults to a timestamp with microseconds. The same values
are used for CSV log and model directories. Set a fresh `run_id` for each new run;
explicitly reusing one can overwrite existing artifacts.

Each run gets a directory such as `lightning_logs/cityscapes/<run_id>/`:

- `metrics.csv` receives loss and IoU values emitted by `self.log(...)`;
- `hparams.yaml` stores the module parameters recorded by `save_hyperparameters()`:
  class count, model width, learning rate, and ignore index (not the full Hydra config).

Training loss is logged per step and per epoch; validation/test loss and IoU
are logged per epoch. CSV rows can have empty cells because metrics arrive at
different times. `training.log_every_n_steps=1` and the logger's flush interval
of one make short demonstrations visible immediately; increase them for longer runs.
This experiment logger is separate from Python's `logging` for diagnostic messages.

To inspect an override without starting training:

```bash
uv run cityscapes-train-lightning --cfg job --resolve \
  experiment.name=seminar experiment.run_id=demo logger=csv
```

### Local MLflow logging

MLflow is installed with the project dependencies. From `cityscapes-tools/`,
start the local tracking server and UI in a separate terminal:

```bash
uv run mlflow server \
  --host 127.0.0.1 --port 5000 --workers 1 \
  --backend-store-uri sqlite:///mlflow.db \
  --serve-artifacts --artifacts-destination ./mlartifacts
```

Open <http://127.0.0.1:5000> in a browser. Run training from another terminal
in the same project directory:

```bash
uv run cityscapes-train-lightning logger=mlflow experiment.name=seminar
```

`src/cityscapes_tools/conf/logger/mlflow.yaml` selects Lightning's supported
`lightning.pytorch.loggers.MLFlowLogger`. The existing runner passes this logger
to `Trainer`; `self.log(...)` and `save_hyperparameters()` supply the metrics and
module hyperparameters through Lightning's normal logging hooks. No direct
`mlflow.start_run()` or `mlflow.log_metric()` calls are needed in the model.

The logger uses `experiment.name` as the MLflow experiment name and
`experiment.run_id` as the displayed run name. MLflow assigns its own internal
run ID and creates a new run on each invocation. It records training loss per
step and epoch, and training/validation/test IoU and validation/test loss per
epoch. Hyperparameters include class count, model width, learning rate, and
ignore index. Writes are synchronous so short demonstration runs are visible
immediately. MLflow is the default logger; select `logger=csv` for CSV files.

Checkpoints continue to be saved under `models/`. To also upload checkpoints
from both configured `ModelCheckpoint` callbacks to MLflow as they are saved:

```bash
uv run cityscapes-train-lightning logger=mlflow logger.log_model=all
```

The server stores run metadata in `mlflow.db` and uploaded artifacts in
`mlartifacts/`. These paths and MLflow's fallback `mlruns/` directory are ignored
by Git. Restart the server from the same directory to see existing runs. Stop
it with Ctrl+C; the stored runs remain on disk. Start the server before training
with this logger. To use another local port, change both `--port` on the server
and `logger.tracking_uri=http://127.0.0.1:<port>` on the training command.

### Best and latest checkpoints

`src/cityscapes_tools/conf/checkpoint/checkpoint.yaml` configures two standard
`ModelCheckpoint` callbacks. Both save full Lightning checkpoints, including
model and optimizer state, under `models/<experiment.name>/<experiment.run_id>/`:

- `best.ckpt`: replaced only when validation IoU improves (`monitor: val_iou`,
  `mode: max`). The name stays exactly `best.ckpt`, without version suffixes.
- `epoch_003.ckpt`: the latest completed epoch; only one numbered checkpoint
  is retained (`monitor: null`, `save_top_k: 1`). Epoch numbering is zero-based,
  so the first completed epoch produces `epoch_000.ckpt`.

The latest checkpoint is saved at training epoch end; the best checkpoint is
selected at validation end. `EarlyStopping` uses the same monitor and direction
as best-checkpoint selection. After fitting, the runner passes the best callback's
explicit file path to `trainer.test`, so it cannot accidentally select the latest
callback's checkpoint. A valid best checkpoint requires completed validation.

For example, after completing epoch 3:

```text
lightning_logs/seminar/<run_id>/
├── metrics.csv
└── hparams.yaml
models/seminar/<run_id>/
├── best.ckpt
└── epoch_003.ckpt
```

Logs and model artifacts are excluded from Git; the source-code package
`src/cityscapes_tools/models/` remains tracked. A checkpoint restores model state;
a full training resume additionally needs a compatible configuration, data,
and `Trainer.fit(..., ckpt_path=...)` (not exposed as a CLI option yet).

## Change the experiment

Hydra can override settings without editing YAML files:

```bash
uv run cityscapes-train-lightning \
  training.batch_size=4 model.base_channels=32
```

For a lasting change, edit the corresponding YAML file. Training outputs such
as project-root `lightning_logs/`, `models/`, and Hydra's `outputs/` are local build
artifacts and are not tracked by Git.

Inspect configuration without starting training:

```bash
uv run cityscapes-train-torch --cfg job
uv run cityscapes-train-lightning --cfg job training.batch_size=4
```

The default YAML files ship in the wheel. Run commands from the directory
containing `data/`; relative dataset paths are resolved from that directory.
Use Hydra's `--config-path /absolute/path/to/conf` with a matching config tree
when supplying external configurations.
