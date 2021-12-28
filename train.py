import argparse

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import get_tinyimagenet
from model import ODENet
from mods import get_modification_transform


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
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss',
            'frequency': 1
        }


def get_run_dir(args):
    if args['modification'] == 'filter':
        op, w = args["operation"], args["window-size"]
        return f'runs/{op}-{w}x{w}'


def main(args):
    seed_everything(42, workers=True)

    run_dir = get_run_dir(args)
    modification = get_modification_transform(**args)
    odenet = LitODENet(modification=modification, lr=1e-2)

    trainer = Trainer(
        default_root_dir=run_dir,
        max_epochs=50,
        gpus=1,
        deterministic=True,
        callbacks=[
            ModelCheckpoint(monitor="val_loss")
        ]
    )

    # trainer.tune(odenet)
    trainer.fit(odenet)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train ODENet Forensic Classifier')
    subparsers = parser.add_subparsers(dest='modification', help='type of image modifications to detect')

    filter_parser = subparsers.add_parser('filter')
    filter_parser.add_argument('operation', choices=('median', 'mean'), help='filter operation')
    filter_parser.add_argument('window-size', type=int, help='filter window size')

    args = parser.parse_args()
    args = vars(args)
    main(args)