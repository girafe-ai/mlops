from dataclasses import dataclass


@dataclass
class Data:
    name: str
    path: str
    test_size: float
    seed: int


@dataclass
class Model:
    name: str
    layers: int
    output_dim: int


@dataclass
class Training:
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    gpu_id: int
    grad_accumulate_batches: int
    precision: str


@dataclass
class Params:
    data: Data
    model: Model
    training: Training
