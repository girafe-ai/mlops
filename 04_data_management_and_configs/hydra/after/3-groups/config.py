from typing import Any
from dataclasses import dataclass


@dataclass
class CIFARData:
    name: str
    path: str
    test_size: float
    seed: int


@dataclass
class MNISTData:
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
    data: Any
    model: Model
    training: Training
