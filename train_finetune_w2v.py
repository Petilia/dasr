import os

import torch
import hydra
from omegaconf import DictConfig
from pathlib import Path

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NUMBA_CACHE_DIR"] = "/tmp/"

import pytorch_lightning as pl

from dasr.train.pl_data import DASRDataModule
from dasr.train.pl_tune_w2v import ASRModel


@hydra.main(version_base=None, config_path="configs", config_name="config_w2v_finetune")
def main(cfg: DictConfig):
    print(cfg)

    torch.set_float32_matmul_precision('medium')

    dm = DASRDataModule(cfg)
    dasr = ASRModel(cfg)

    loggers = [
        pl.loggers.WandbLogger(
            project=cfg.wandb.project_name,
            name=cfg.wandb.run_name,
        ),
    ]


    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(
            max_depth=1
        ),
    ]

    if cfg.artifacts.checkpoint.use:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=Path(cfg.artifacts.checkpoint.dirpath)
            / cfg.wandb.run_name,
            filename=cfg.artifacts.checkpoint.filename,
            monitor=cfg.artifacts.checkpoint.monitor,
            mode=cfg.artifacts.checkpoint.mode,
            save_top_k=cfg.artifacts.checkpoint.save_top_k,
            every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
            verbose=True,
        )

        callbacks.append(checkpoint_callback)


    trainer = pl.Trainer(accelerator=cfg.train.accelerator, 
                         devices=cfg.train.devices, 
                         log_every_n_steps=cfg.train.log_every_n_steps, 
                         gradient_clip_val=cfg.train.gradient_clip_val,
                         precision=cfg.train.precision,
                         enable_checkpointing=cfg.artifacts.checkpoint.use,
                         callbacks=callbacks,
                         logger=loggers, 
                         strategy='ddp_find_unused_parameters_true',
                         )
    
    trainer.fit(model=dasr, datamodule=dm)


if __name__ == "__main__":
    main()
