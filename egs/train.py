import os
import sys
import json
import pprint
sys.path.append("../src")
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from criterion.sdr import ClippedNegSISDR, NegSISDR
# from models.sepformer import SepFormer
from models.conv_tasnet import ConvTasNet
from models.tse_sepformer import SepFormer
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from datasets import PTSEDataset


class System(pl.LightningModule):
    def __init__(
            self, 
            args,
            model,
            loss_func,
            optimizer,
            train_loader,
            val_loader,
        ):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        seed_everything(args.seed)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def forward(self, batch):
        x, e = batch["noisy"], batch["spk_emb"]
        return self.model(x, e)

    def training_step(self, batch, batch_idx):
        y, x, e = batch["clean_wave"], batch["noisy_wave"], batch["spk_emb"]
        # y_hat = self.model(x)
        y_hat = self.model(x, e)
        loss = self.loss_func(y_hat, y)
        self.log(
            "train_loss", loss, 
            on_epoch=True, logger=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        y, x, e = batch["clean_wave"], batch["noisy_wave"], batch["spk_emb"]
        # y_hat = self.model(x)
        y_hat = self.model(x, e)
        loss = self.loss_func(y_hat, y)
        self.log(
            "val_loss", loss, 
            on_epoch=True, logger=True, sync_dist=True
        )

    def configure_optimizers(self):
        # optimizer = self.optimizer(self.model.parameters(), lr=self.args.lr)
        if self.args.half_lr:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=self.optimizer, factor=0.5, patience=5
            )
            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "epoch",
                    "strict": True,
                    "name": "current_scheduled_lr",
                }
            }
        else:
            return self.optimizer


def main(args):
    
    train_set = PTSEDataset(
        csv_dir=args.train_csv_dir,
        seg_len=args.seg_len,
        sample_rate=args.sample_rate,
    )
    val_set = PTSEDataset(
        csv_dir=args.val_csv_dir,
        seg_len=args.seg_len,
        sample_rate=args.sample_rate,
    )
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True,
    )

    model = SepFormer(
        args.n_basis, args.kernel_size, stride=args.stride,
        enc_basis=args.enc_basis, dec_basis=args.dec_basis, enc_nonlinear=args.enc_nonlinear,
        window_fn=args.window_fn, enc_onesided=args.enc_onesided, enc_return_complex=args.enc_return_complex,
        sep_bottleneck_channels=args.sep_bottleneck_channels,
        sep_chunk_size=args.sep_chunk_size, sep_hop_size=args.sep_hop_size,
        sep_num_blocks=args.sep_num_blocks,
        sep_num_layers_intra=args.sep_num_layers_intra, sep_num_layers_inter=args.sep_num_layers_inter,
        sep_num_heads_intra=args.sep_num_heads_intra, sep_num_heads_inter=args.sep_num_heads_inter,
        sep_d_ff_intra=args.sep_d_ff_intra, sep_d_ff_inter=args.sep_d_ff_inter,
        sep_norm=args.sep_norm, sep_nonlinear=args.sep_nonlinear, sep_dropout=args.sep_dropout, mask_nonlinear=args.mask_nonlinear,
        causal=args.causal,
        n_sources=args.n_sources,
    )
    # model = ConvTasNet(
    #     args.n_basis, args.kernel_size, stride=args.stride, enc_basis=args.enc_basis, dec_basis=args.dec_basis, enc_nonlinear=args.enc_nonlinear,
    #     window_fn=args.window_fn, enc_onesided=args.enc_onesided, enc_return_complex=args.enc_return_complex,
    #     sep_hidden_channels=args.sep_hidden_channels, sep_bottleneck_channels=args.sep_bottleneck_channels, sep_skip_channels=args.sep_skip_channels,
    #     sep_kernel_size=args.sep_kernel_size, sep_num_blocks=args.sep_num_blocks, sep_num_layers=args.sep_num_layers,
    #     dilated=args.dilated, separable=args.separable, causal=args.causal, sep_nonlinear=args.sep_nonlinear, sep_norm=args.sep_norm, mask_nonlinear=args.mask_nonlinear,
    #     n_sources=args.n_sources
    # )
    print("# Parameters: {}".format(model.num_parameters))
    loss_func = NegSISDR()
    # loss_func = nn.MSELoss()
    optim_dict = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
    }
    optimizer = optim_dict[args.optimizer](model.parameters(), lr=args.lr)
    system = System(
        args=args,
        model=model,
        optimizer=optimizer,
        loss_func=loss_func,
        train_loader=train_loader,
        val_loader=val_loader,
        # optimizer=optim_dict[args.optimizer],
    )

    callbacks = []
    checkpoint = ModelCheckpoint(
        args.model_dir, monitor="val_loss", mode="min", save_top_k=1, verbose=True
    )
    callbacks.append(checkpoint)
    if args.early_stop:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, verbose=True))
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=args.log_dir)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        amp_backend="native",
        logger=tb_logger,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
        max_epochs=args.epochs,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.max_norm,
        resume_from_checkpoint=args.continue_from,
        track_grad_norm=args.track_grad_norm,
        precision=args.precision,
    )
    trainer.fit(system)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training of SepFormer")

    parser.add_argument('--cfg', type=str)
    parser.add_argument('--train_csv_dir', type=str, default=None, help='Path for training dataset csv ROOT directory')
    parser.add_argument('--val_csv_dir', type=str, default=None, help='Path for validation dataset csv ROOT directory')
    parser.add_argument('--sample_rate', '-sr', type=int, default=16000, help='Sampling rate')
    parser.add_argument("--seg_len", type=float, default=10.0, help="Length of truncated input in seconds")
    parser.add_argument('--enc_basis', type=str, default='trainable', choices=['trainable','Fourier','trainableFourier','trainableFourierTrainablePhase'], help='Encoder type')
    parser.add_argument('--dec_basis', type=str, default='trainable', choices=['trainable','Fourier','trainableFourier','trainableFourierTrainablePhase', 'pinv'], help='Decoder type')
    parser.add_argument('--enc_nonlinear', type=str, default=None, help='Non-linear function of encoder')
    parser.add_argument('--window_fn', type=str, default='hann', help='Window function')
    parser.add_argument('--enc_onesided', type=int, default=None, choices=[0, 1, None], help='If true, encoder returns kernel_size // 2 + 1 bins.')
    parser.add_argument('--enc_return_complex', type=int, default=None, choices=[0, 1, None], help='If true, encoder returns complex tensor, otherwise real tensor concatenated real and imaginary part in feature dimension.')
    parser.add_argument('--n_basis', '-F', type=int, default=256, help='# basis')
    parser.add_argument('--kernel_size', '-L', type=int, default=2, help='Kernel size')
    parser.add_argument('--stride', type=int, default=None, help='Stride. If None, stride=kernel_size // 2')
    parser.add_argument('--sep_bottleneck_channels', '-B', type=int, default=None, help='Bottleneck channels of separator')
    parser.add_argument('--sep_chunk_size', '-C', type=int, default=250, help='Chunk size of separator')
    parser.add_argument('--sep_hop_size', '-P', type=int, default=125, help='Hop size of separator')
    parser.add_argument('--sep_num_blocks', '-N', type=int, default=2, help='# blocks of separator.')
    parser.add_argument('--sep_num_layers_intra', '-K_intra', type=int, default=8, help='# layers of intra transformer.')
    parser.add_argument('--sep_num_layers_inter', '-K_inter', type=int, default=8, help='# layers of inter transformer.')
    parser.add_argument('--sep_num_heads_intra', '-h_intra', type=int, default=8, help='# heads of intra transformer.')
    parser.add_argument('--sep_num_heads_inter', '-h_inter', type=int, default=8, help='# heads of inter transformer.')
    parser.add_argument('--sep_d_ff_intra', '-d_ff_intra', type=int, default=1024, help='# dimensions of feedforward module in intra transformer.')
    parser.add_argument('--sep_d_ff_inter', '-d_ff_inter', type=int, default=1024, help='# dimensions of feedforward module in inter transformer.')
    parser.add_argument('--causal', type=int, default=0, help='Causality')
    parser.add_argument('--sep_norm', type=int, default=1, help='Normalization')
    parser.add_argument('--sep_nonlinear', type=str, default='relu', help='Non-linear function of separator')
    parser.add_argument('--sep_dropout', type=float, default=0, help='Dropout')
    parser.add_argument('--mask_nonlinear', type=str, default='sigmoid', help='Non-linear function of mask estiamtion')
    parser.add_argument('--n_sources', type=int, default=None, help='# speakers')
    parser.add_argument('--criterion', type=str, default='clipped-sisdr', choices=['clipped-sisdr', 'sisdr'], help='Criterion')
    parser.add_argument('--clip', type=float, default=30, help='Clip of SI-SDR.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'rmsprop'], help='Optimizer, [sgd, adam, rmsprop]')
    parser.add_argument('--lr', type=float, default=15e-5, help='Learning rate during warm up. Default: 15e-5')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay (L2 penalty). Default: 0')
    parser.add_argument('--max_norm', type=float, default=None, help='Gradient clipping')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size. Default: 4')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--model_dir', type=str, default='./tmp/model', help='Model directory')
    parser.add_argument('--loss_dir', type=str, default='./tmp/loss', help='Loss directory')
    parser.add_argument('--sample_dir', type=str, default='./tmp/sample', help='Sample directory')
    parser.add_argument('--continue_from', type=str, default=None, help='Resume training')
    parser.add_argument('--use_cuda', type=int, default=1, help='0: Not use cuda, 1: Use cuda')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--track_grad_norm', type=int, default=-1, help='Track gradient p-norm. Default -1 without tracking.')

    args = parser.parse_args()
    with open(args.cfg, "r") as f:
        args = parser.parse_args(namespace=argparse.Namespace(**json.load(f)))

    if not os.path.exists(args.exp_dir):
        os.makedirs(args.exp_dir)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.config_dir):
        os.makedirs(args.config_dir)

    # with open(os.path.join(args.config_dir, "config.json"), "w") as f:
    #     config = vars(args)
    #     json.dump(config, f)

    main(args)