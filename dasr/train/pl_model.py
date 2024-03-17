from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate


class DASRModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.asr = instantiate(cfg.asr)
        self.model = instantiate(cfg.denoiser)

        # define loss
        self.geom_loss = instantiate(cfg.loss)
        # self.geom_loss.to(self.device)

        self.use_asr_loss = cfg.train.use_asr_loss
        self.use_geom_loss = cfg.train.use_geom_loss

        self.n_epoch_before_asr_loss = cfg.train.n_epoch_before_asr_loss
        self.asr_loss_coef = cfg.train.asr_loss_coef
        # self.only_asr_loss = cfg.train.only_asr_loss
        if not self.use_geom_loss:
            self.n_epoch_before_asr_loss = 0
            self.asr_loss_coef = 1

    def forward(self, batch):
        noisy = batch["noise_audios"]
        output = self.model(noisy)
        output = output.squeeze(1)
        return output

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        output = self(batch)

        gt_transcript = batch["transcriptions"]
        clear = batch["clean_audios"]
        noisy = batch["noise_audios"]
        # attention_mask = batch["noise_attention_masks"]
        attention_mask = None

        if self.use_geom_loss:
            geom_loss, geom_loss_stats = self.geom_loss(clear, output)
            geom_loss_stats = {
                f"train/{key}": value for key, value in geom_loss_stats.items()
            }
            self.log_dict(geom_loss_stats, on_step=True, on_epoch=False, prog_bar=True)
        else:
            geom_loss = 0

        if self.use_asr_loss and self.current_epoch >= self.n_epoch_before_asr_loss:
            asr_loss, asr_loss_stats = self.asr.get_loss(
                clear,
                output,
                noisy_speech=noisy,
                gt_transcript=gt_transcript,
                attention_mask=attention_mask,
            )
            asr_loss_stats = {
                f"train/{key}": value for key, value in asr_loss_stats.items()
            }
            self.log_dict(asr_loss_stats, on_step=True, on_epoch=False, prog_bar=True)

        else:
            asr_loss = 0

        loss = geom_loss + self.asr_loss_coef * asr_loss

        # self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        output = self(batch)

        gt_transcript = batch["transcriptions"]
        clear = batch["clean_audios"]
        noisy = batch["noise_audios"]
        # attention_mask = batch["noise_attention_masks"]
        attention_mask = None

        if self.use_geom_loss:
            geom_loss, geom_loss_stats = self.geom_loss(clear, output)
            geom_loss_stats = {
                f"val/{key}": value for key, value in geom_loss_stats.items()
            }
            self.log_dict(geom_loss_stats, on_step=False, on_epoch=True, prog_bar=True)
        else:
            geom_loss = 0

        if self.use_asr_loss and self.current_epoch >= self.n_epoch_before_asr_loss:
            asr_loss, asr_loss_stats = self.asr.get_loss(
                clear,
                output,
                noisy_speech=noisy,
                gt_transcript=gt_transcript,
                attention_mask=attention_mask,
            )
            asr_loss_stats = {
                f"val/{key}": value for key, value in asr_loss_stats.items()
            }
            self.log_dict(asr_loss_stats, on_step=False, on_epoch=True, prog_bar=True)
        else:
            asr_loss = 0

        loss = geom_loss + self.asr_loss_coef * asr_loss

        # self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        return {"val_loss": loss}

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.train.optimizer.learning_rate,
        )
        return {"optimizer": optimizer}

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
