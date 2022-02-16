import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import get_tinyimagenet
from model import ODENet
from mods import add_modification_argparse, get_modification_string, get_modification_transform


class LitODENet(pl.LightningModule):
    def __init__(self, modification=None, lr=1e-3, batch_size=16):
        super().__init__()
        self.model = ODENet()
        self.lr = lr
        self.batch_size = batch_size

        self.modif_transform = modification
    
    def train_dataloader(self):
        train_dataset = get_tinyimagenet(self.modif_transform, num_images=2_000, split='train')
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        val_dataset = get_tinyimagenet(self.modif_transform, num_images=2_000, split='val')
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=8)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze(dim=1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        accuracy = ((y_hat >= 0.5) == y).float().mean()
        self.log('train_loss', loss, on_epoch=True)
        self.log('train_accuracy', accuracy, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze(dim=1)
        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        accuracy = ((y_hat >= 0.5) == y).float().mean()
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)

        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
                'monitor': 'val_loss',
                'frequency': 1
            }
        }


def main(args):
    seed_everything(42, workers=True)

    run_dir = 'runs/' + get_modification_string(args)
    modification = get_modification_transform(args)
    odenet = LitODENet(modification=modification, lr=1e-3)

    resume = None
    if args.get('resume', False):
        ckpts = Path(run_dir).glob('lightning_logs/version_*/checkpoints/*.ckpt')
        ckpts = sorted(ckpts, reverse=True, key=lambda p: p.stat().st_mtime)
        resume = ckpts[0] if ckpts else None

    trainer = Trainer(
        resume_from_checkpoint=resume,
        default_root_dir=run_dir,
        max_epochs=args['epochs'],
        gpus=1,
        deterministic=True,
        callbacks=[
            ModelCheckpoint(monitor="val_loss", save_last=True),
            LearningRateMonitor(logging_interval='step')
        ]
    )

    trainer.fit(odenet)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ODENet Forensic Classifier')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of training epochs')
    parser.add_argument('-r', '--resume', default=False, action='store_true', help='resume training')

    add_modification_argparse(parser)

    args = parser.parse_args()
    args = vars(args)
    main(args)