import os
import time

from omegaconf import DictConfig, OmegaConf, open_dict

from ppo.infrastructure.rl_trainer import RL_Trainer


def my_app(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    logdir_prefix = "ppo_"  # keep for autograder
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../data")
    env_name = cfg.env.env_name.replace("/", "-")  # MinAtar/SpaceInvaders-v1 -> MinAtar-SpaceInvaders-v1
    exp_name = logdir_prefix + cfg.env.exp_name + "_" + env_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, exp_name)
    os.makedirs(logdir, exist_ok=True)
    with open_dict(cfg):
        cfg.logging.logdir = logdir
    OmegaConf.save(cfg, os.path.join(logdir, "config.yaml"))
    print("\nLOGGING TO:", logdir, "\n")

    trainer = RL_Trainer(cfg)
    return trainer.run_training_loop()
