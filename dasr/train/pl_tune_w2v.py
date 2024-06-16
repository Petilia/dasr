from typing import Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate


class ASRModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        
        self.asr = instantiate(cfg.asr)


    def forward(self, batch):
        # forward through denoiser
        noisy = batch["noise_audios"]
       
        # forward through asr
        attention_mask = batch["noise_attention_masks"]
        gt_transcript = batch["transcriptions"]
        asr_output = self.asr.forward(noisy, attention_mask, gt_transcript)

        return asr_output, noisy

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        asr_output, _ = self(batch)

        gt_transcript = batch["transcriptions"]
        clear = batch["clean_audios"]
        noisy = batch["noise_audios"]
        # attention_mask = batch["noise_attention_masks"]
        attention_mask = None
     
        loss, loss_stats = self.asr.get_loss(
            asr_output,
            clear,
            attention_mask=attention_mask,
            noisy_speech=noisy,
            gt_transcript=gt_transcript,
        )
        loss_stats = {
            f"train/{key}": value for key, value in loss_stats.items()
        }
        self.log_dict(loss_stats, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True, batch_size=len(batch["transcriptions"]))
      
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx=0):
        asr_output, _ = self(batch)

        gt_transcript = batch["transcriptions"]
        clear = batch["clean_audios"]
        noisy = batch["noise_audios"]
        # attention_mask = batch["noise_attention_masks"]
        attention_mask = None

        loss, loss_stats = self.asr.get_loss(
            asr_output,
            clear,
            attention_mask=attention_mask,
            noisy_speech=noisy,
            gt_transcript=gt_transcript,
        )
        loss_stats = {
            f"val/{key}": value for key, value in loss_stats.items()
        }
        self.log_dict(loss_stats, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch["transcriptions"]))


    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(
            self.asr.parameters(),
            lr=self.cfg.train.optimizer.learning_rate,
        )
        return {"optimizer": optimizer}

    def on_before_optimizer_step(self, optimizer):
        self.log_dict(pl.utilities.grad_norm(self, norm_type=2))
        super().on_before_optimizer_step(optimizer)
