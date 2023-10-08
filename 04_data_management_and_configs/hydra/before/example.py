from typing import Dict, Any


config = {
    "data": {
        "name": "cifar",
        "path": "/data/to/cifar/dataset",
        "test_size": 0.2,
        "seed": 42,
    },
    "model": {
        "name": "resnet",
        "layers": 50,
        "output_dim": 128,
    },
    "training": {
        "batch_size": 32,
        "epochs": 10,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "gpu_id": 2,
        "grad_accumulate_batches": 4,
        "precision": "fp16",
    },
}


def main(cfg: Dict[str, Any]) -> None:
    print(cfg)


if __name__ == "__main__":
    main(config)
