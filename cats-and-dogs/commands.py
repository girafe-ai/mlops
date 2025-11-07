import fire

from cats_and_dogs.infer import infer  # noqa: F401
from cats_and_dogs.train import train  # noqa: F401

if __name__ == "__main__":
    fire.Fire()
