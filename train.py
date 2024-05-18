import os

import torch
import hydra
from omegaconf import DictConfig

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/"

import pytorch_lightning as pl

from dasr.train.pl_data import DASRDataModule
from dasr.train.pl_model import DASRModel


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(cfg)

    torch.set_float32_matmul_precision('medium')

    dm = DASRDataModule(cfg)
    dasr = DASRModel(cfg)

    loggers = [
        pl.loggers.WandbLogger(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
        ),
    ]

    trainer = pl.Trainer(accelerator="cuda", logger=loggers, log_every_n_steps=15)
    trainer.fit(model=dasr, datamodule=dm)


if __name__ == "__main__":
    main()
