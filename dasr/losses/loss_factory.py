import torch
import torch.nn.functional as F

from .sfft_loss import MultiResolutionSTFTLoss


class AdditiveLoss(torch.nn.Module):
    def __init__(
        self, criteria="l1", use_sfft_loss=True, factor_sc=0.5, factor_mag=0.5
    ):
        super().__init__()
        self.sfft_loss = MultiResolutionSTFTLoss(
            factor_sc=factor_sc, factor_mag=factor_mag
        )
        self.criteria = criteria
        self.use_sfft_loss = use_sfft_loss

    def forward(self, clean, estimate):
        loss_stats = {}
        with torch.autograd.set_detect_anomaly(True):
            if self.criteria == "l1":
                loss = F.l1_loss(clean, estimate)
            elif self.criteria == "l2":
                loss = F.mse_loss(clean, estimate)
            elif self.criteria == "huber":
                loss = F.smooth_l1_loss(clean, estimate)
            else:
                raise ValueError(f"Invalid loss {self.criteria}")
            loss_stats[self.criteria] = loss.item()
            # MultiResolution STFT loss
            if self.use_sfft_loss:
                sc_loss, mag_loss = self.sfft_loss(
                    estimate.squeeze(1), clean.squeeze(1)
                )
                loss += sc_loss + mag_loss
                loss_stats["sc_loss"] = sc_loss.item()
                loss_stats["mag_loss"] = mag_loss.item()
            loss_stats["total_loss"] = loss.item()

            # TODO добавить возвращение словаря со статистиками по всем лоссам
            return loss, loss_stats
