import hydra
from hydra.core.config_store import ConfigStore

from config import Params


cs = ConfigStore.instance()
cs.store(name="params", node=Params)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: Params) -> None:
    print(cfg)


if __name__ == "__main__":
    main()
