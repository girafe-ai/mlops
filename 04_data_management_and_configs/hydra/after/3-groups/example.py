import hydra
from hydra.core.config_store import ConfigStore

from config import Params, CIFARData, MNISTData


cs = ConfigStore.instance()
cs.store(name="params", node=Params)
cs.store(group="data", name="base_cifar", node=CIFARData)
cs.store(group="data", name="base_mnist", node=MNISTData)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    print(dir(cfg))
    print(cfg)


# python example.py +data=cifar
# python example.py +data=mnist
if __name__ == "__main__":
    main()
