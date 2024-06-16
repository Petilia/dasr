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
        
        self.use_asr_loss = cfg.train.use_asr_loss
        self.use_geom_loss = cfg.train.use_geom_loss

        self.n_epoch_before_asr_loss = cfg.train.n_epoch_before_asr_loss
        self.asr_loss_coef = cfg.train.asr_loss_coef
    
        if not self.use_geom_loss:
            self.n_epoch_before_asr_loss = 0
            self.asr_loss_coef = 1

    def forward(self, batch):
        # forward through denoiser
        noisy = batch["noise_audios"]
        denoisy_speech = self.model(noisy)
        denoisy_speech = denoisy_speech.squeeze(1)

        # forward through asr
        attention_mask = batch["noise_attention_masks"]
        gt_transcript = batch["transcriptions"]
        
        asr_output = self.asr.forward(denoisy_speech, attention_mask, gt_transcript)

        return asr_output, denoisy_speech

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        asr_output, denoisy_speech = self(batch)

        gt_transcript = batch["transcriptions"]
        clear = batch["clean_audios"]
        noisy = batch["noise_audios"]
        # attention_mask = batch["noise_attention_masks"]
        attention_mask = None

        if self.use_geom_loss:
            geom_loss, geom_loss_stats = self.geom_loss(clear, denoisy_speech)
            geom_loss_stats = {
                f"train/{key}": value for key, value in geom_loss_stats.items()
            }
            self.log_dict(geom_loss_stats, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, batch_size=len(batch["transcriptions"]))
        else:
            geom_loss = 0

        if self.use_asr_loss and self.current_epoch >= self.n_epoch_before_asr_loss:
            asr_loss, asr_loss_stats = self.asr.get_loss(
                asr_output,
                clear,
                attention_mask=attention_mask,
                noisy_speech=noisy,
                gt_transcript=gt_transcript,
            )
            asr_loss_stats = {
                f"train/{key}": value for key, value in asr_loss_stats.items()
            }
            self.log_dict(asr_loss_stats, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, batch_size=len(batch["transcriptions"]))
        else:
            asr_loss = 0

        loss = geom_loss + self.asr_loss_coef * asr_loss

        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        asr_output, denoisy_speech = self(batch)

        gt_transcript = batch["transcriptions"]
        clear = batch["clean_audios"]
        noisy = batch["noise_audios"]
        # attention_mask = batch["noise_attention_masks"]
        attention_mask = None

        # if self.use_geom_loss:
        # На валидации смотрим на оба лосса без if-ов
        geom_loss, geom_loss_stats = self.geom_loss(clear, denoisy_speech)
        geom_loss_stats = {
            f"val/{key}": value for key, value in geom_loss_stats.items()
        }
        self.log_dict(geom_loss_stats, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch["transcriptions"]))
        # else:
        #     geom_loss = 0

        # if self.use_asr_loss and self.current_epoch >= self.n_epoch_before_asr_loss:
        asr_loss, asr_loss_stats = self.asr.get_loss(
            asr_output,
            clear,
            attention_mask=attention_mask,
            noisy_speech=noisy,
            gt_transcript=gt_transcript,
        )
        asr_loss_stats = {
            f"val/{key}": value for key, value in asr_loss_stats.items()
        }
        self.log_dict(asr_loss_stats, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch["transcriptions"]))
        # else:
        #     asr_loss = 0

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
