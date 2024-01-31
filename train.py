import os

import hydra
from omegaconf import DictConfig

from dasr.train.trainer import Trainer

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/"


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(cfg)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
