import hydra
from omegaconf import DictConfig

from ppo.main import my_app


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_main(cfg: DictConfig):
    my_app(cfg)


if __name__ == "__main__":
    my_main()
