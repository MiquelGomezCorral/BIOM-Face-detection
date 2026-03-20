import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import torchvision.transforms as transforms


import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from src.config import Configuration
from src.data import FACES_DATASET



class FaceCNN(nn.Module):
    def __init__(self, in_channels, num_classes, out_size=(1, 1)):
        super(FaceCNN, self).__init__()
        self.out_size = out_size
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding="same")
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding="same")
        self.bn2   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same")
        self.bn3   = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same")
        self.bn4   = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same")
        self.bn5   = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * self.out_size[0] * self.out_size[1], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = F.adaptive_avg_pool2d(x, output_size=self.out_size)
        x = torch.flatten(x, 1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x  # raw logits — use BCEWithLogitsLoss
    


class FaceDetectionModule(pl.LightningModule):
    def __init__(self, CONFIG: Configuration, model: nn.Module = None):
        super().__init__()
        self.config = CONFIG
        self.model = model if model is not None else FaceCNN(
            in_channels=1 if CONFIG.gray_scale else 3,
            num_classes=1,
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc   = BinaryAccuracy()
        self.test_acc  = BinaryAccuracy()
        self.val_f1    = BinaryF1Score()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x     = batch["img"].float()
        y     = batch["label"].float().unsqueeze(1)
        y_hat = self(x)
        loss  = self.criterion(y_hat, y)
        preds = (y_hat > 0.0).squeeze(1)  # threshold at 0 for logits
        return loss, preds, batch["label"]

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.train_acc(preds, labels)
        self.log("train_loss", loss,           prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc",  self.train_acc,  prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log("val_loss", loss,          prog_bar=True)
        self.log("val_acc",  self.val_acc,   prog_bar=True)
        self.log("val_f1",   self.val_f1,    prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        self.log("test_loss", loss)
        self.log("test_acc",  self.test_acc)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class FaceDataModule(pl.LightningDataModule):
    def __init__(self, CONFIG: Configuration):
        super().__init__()
        self.config = CONFIG

    def setup(self, stage=None):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=self.config.aug_prob),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ], p=self.config.aug_prob),
            transforms.ToTensor(),
        ])
        base_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train_ds = FACES_DATASET("train", transform=train_transform, CONFIG=self.config)
        self.val_ds   = FACES_DATASET("val",   transform=base_transform,  CONFIG=self.config)
        self.test_ds  = FACES_DATASET("test",  transform=base_transform,  CONFIG=self.config)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers, pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.config.batch_size, shuffle=False,
            num_workers=self.config.num_workers, pin_memory=True,
            persistent_workers=self.config.num_workers > 0,
        )
