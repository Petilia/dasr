from typing import Optional

import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data import DataLoader


class DASRDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.collate_fn = instantiate(self.cfg.data.collate_class).collate_fn

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.train_dataset = instantiate(self.cfg.data.train_dataset)
            self.val_dataset = instantiate(self.cfg.data.val_dataset)

        if stage == "predict":
            # Сюда просто instantiate(cfg.dataset)
            self.test_dataset = None

    def train_dataloader(self):
        # Здесь взять self.train_dataset и пильнуть dataloader
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.train_batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            num_workers=5,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.data.val_batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
            num_workers=5,
        )

    def test_dataloader(self):
        pass
