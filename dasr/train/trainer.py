import os
from datetime import datetime
from pathlib import Path

import torch
from hydra.utils import instantiate
from tqdm import tqdm

from .utils import sum_list_dicts


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.model = instantiate(cfg.denoiser)
        assert (
            self.model(torch.randn(1, 3000)).shape[-1] == torch.randn(1, 3000).shape[-1]
        ), "input_dim != output_dim"

        self.model.to(self.device)
        self.optimizer = instantiate(
            cfg.train.optimizer, params=self.model.parameters()
        )

        self.asr = instantiate(cfg.asr)
        self.asr_metric = cfg.asr.asr_metric

        self.train_loader, self.test_loader = instantiate(cfg.data)

        # define loss
        self.add_loss = instantiate(cfg.loss)
        self.add_loss.to(self.device)

        self.n_epoch_before_asr_loss = cfg.train.n_epoch_before_asr_loss
        self.asr_loss_coef = cfg.train.asr_loss_coef
        self.only_asr_loss = cfg.train.only_asr_loss
        if self.only_asr_loss:
            self.n_epoch_before_asr_loss = 0

        self.logger = instantiate(cfg.wandb)
        self.checkpoints_dir = Path(
            f"checkpoints/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{cfg.wandb.run_name}"
        )
        os.makedirs(self.checkpoints_dir)

        self.n_epoch = cfg.train.n_epoch
        self.epoch = 0
        self.step = 0

    def train_epoch(self):
        self.model.train()
        self.logger.set_mode("train")

        n_ep_it_loss = 50
        stats = []
        asr_stats = []
        for i, batch in tqdm(enumerate(self.train_loader)):
            self.optimizer.zero_grad()
            gt_transcript = batch["transcriptions"]
            clear = batch["clean_audios"].to(self.device)
            noisy = batch["noise_audios"].to(self.device)
            output = self.model(noisy)
            output = output.squeeze(1)

            if self.epoch >= self.n_epoch_before_asr_loss:
                asr_loss, asr_loss_stats = self.asr.get_loss(
                    clear, output, noisy_speech=noisy, gt_transcript=gt_transcript
                )
                if not self.only_asr_loss:
                    asr_loss *= self.asr_loss_coef
                    asr_loss.backward(retain_graph=True)
                else:
                    asr_loss.backward()

                asr_stats.append(asr_loss_stats)

            if not self.only_asr_loss:
                loss, loss_stats = self.add_loss(clear, output)
                loss.backward()
                stats.append(loss_stats)

            self.optimizer.step()

            if i % n_ep_it_loss == n_ep_it_loss - 1:
                stats = sum_list_dicts(stats)
                if self.epoch >= self.n_epoch_before_asr_loss:
                    asr_stats = sum_list_dicts(asr_stats)
                    stats = stats | asr_stats
                stats["epoch"] = self.epoch
                self.logger.log_dict(stats)
                stats = []
                asr_stats = []

            self.step += 1
            self.logger.set_step(self.step)

    def eval_epoch(self):
        self.model.eval()
        self.logger.set_mode("val")

        stats = []
        asr_stats = []
        for batch in tqdm(self.test_loader):
            gt_transcript = batch["transcriptions"]
            clear = batch["clean_audios"].to(self.device)
            noisy = batch["noise_audios"].to(self.device)
            with torch.no_grad():
                output = self.model(noisy)

            output = output.squeeze(1)
            _, loss_stats = self.add_loss(clear, output)
            asr_loss_stats = self.asr.eval(
                clear, output, noisy_speech=noisy, gt_transcript=gt_transcript
            )
            # print(asr_loss_stats)

            stats.append(loss_stats)
            asr_stats.append(asr_loss_stats)

        stats = sum_list_dicts(stats)
        asr_stats = sum_list_dicts(asr_stats)
        stats = stats | asr_stats
        stats["epoch"] = self.epoch
        self.logger.log_dict(stats)
        return stats

    def eval_iter(self):
        self.model.eval()
        self.logger.set_mode("val")

        batch = next(iter(self.test_loader))
        gt_transcript = batch["transcriptions"]
        clear = batch["clean_audios"].to(self.device)
        noisy = batch["noise_audios"].to(self.device)

        with torch.no_grad():
            output = self.model(noisy)

        output = output.squeeze(1)
        asr_loss_stats = self.asr.eval(
            clear, output, noisy_speech=noisy, gt_transcript=gt_transcript
        )
        return asr_loss_stats

    def train(self):
        best_val_metric = 1e6
        for i in range(self.n_epoch):
            self.train_epoch()
            torch.cuda.empty_cache()
            eval_stats = self.eval_epoch()
            torch.cuda.empty_cache()
            if eval_stats[f"{self.asr_metric} (ref-denoisy)"] < best_val_metric:
                best_val_metric = eval_stats[f"{self.asr_metric} (ref-denoisy)"]
                self.save_weights(best=best_val_metric)
            else:
                self.save_weights()
            self.epoch += 1

    def save_weights(self, best=False):
        checkpoint_dict = {
            "epoch": self.epoch,
            "config": self.cfg,
            # "stats_dict": stats_dict,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        torch.save(checkpoint_dict, self.checkpoints_dir / f"epoch_{self.epoch}.pth")

        if best:
            torch.save(checkpoint_dict, self.checkpoints_dir / "best.pth")
            print(f"Metric improved to {best}")
            wandb_log_path = str((self.checkpoints_dir / "best.pth").relative_to("."))
            self.logger.log_best_model(wandb_log_path)
