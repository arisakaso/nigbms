# %%
import hydra
import pandas as pd
import torch
import wandb
from hydra.utils import instantiate
from lightning import LightningModule, seed_everything
from omegaconf import DictConfig
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger


# %%
class NIGBMS(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.solver = instantiate(cfg.solver)
        self.surrogate = instantiate(cfg.surrogate)
        self.wrapper = instantiate(cfg.wrapper, solver=self.solver, surrogate=self.surrogate)

    def on_fit_start(self):
        seed_everything(seed=self.hparams.train.seed, workers=True)

    def forward(self, tau):
        theta = self.meta_solver(tau)
        history = self.solver(tau, theta)
        losses = self.loss(tau, theta, history)
        return losses

    def _add_prefix(self, d, prefix):
        return dict([(prefix + k, v) for k, v in d.items()])

    def training_step(self, batch, batch_idx):
        tau = batch
        losses = self(tau)
        self.log_dict(self._add_prefix(losses, "train/loss/"), logger=True, on_epoch=True, on_step=False)

        return losses["combined"]

    def on_validation_epoch_start(self) -> None:
        self.maxiter_train = self.hparams.solver.opts_fixed.maxiter
        self.hparams.solver.opts_fixed.maxiter = 100_000
        self.solver = PytorchLinearSolverModule(**self.hparams.solver)

    def validation_step(self, batch, batch_idx):
        tau = batch
        losses = self(tau)
        self.log_dict(self._add_prefix(losses, "val/loss/"), logger=True, on_epoch=True, on_step=False)

        return losses["combined"]

    def on_validation_epoch_end(self) -> None:
        self.hparams.solver.opts_fixed.maxiter = self.maxiter_train
        self.solver = PytorchLinearSolverModule(**self.hparams.solver)

    def on_test_epoch_start(self) -> None:
        # test settings
        self.hparams.solver.opts_fixed.maxiter = 100_000
        self.solver = PytorchLinearSolverModule(**self.hparams.solver)

    def test_step(self, batch, batch_idx, dataloader_idx):
        tau = batch
        losses = self(tau)
        self.log_dict(self._add_prefix(losses, "test/loss/"), logger=True, on_epoch=True, on_step=False)

        # non-learning baseline
        theta = self.meta_solver(tau)
        theta["initial_guess"] = torch.zeros_like(theta["initial_guess"])
        history = self.solver(tau, theta)
        losses = self.loss(tau, theta, history)
        self.log_dict(self._add_prefix(losses, "baseline/loss/"), logger=True, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.train.optimizer, params=self.parameters())
        scheduler = instantiate(self.hparams.train.lr_scheduler, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": self.hparams.train.monitor,
        }


@hydra.main(config_path="../configs", config_name="train_poisson1d", version_base=None)
def main(cfg: DictConfig):
    # FIX SEED
    pl.seed_everything(seed=cfg.train.seed, workers=True)

    torch.set_default_dtype(torch.float64)

    # MODEL SETUP
    if cfg.train.ckpt_path:
        gbms = GBMSPoisson1D.load_from_checkpoint(cfg.train.ckpt_path, cfg=cfg)
    else:
        gbms = GBMSPoisson1D(cfg=cfg)

    # DATA SETUP
    meta_df = pd.read_csv(cfg.data.root_dir + "/meta_df.csv")
    meta_dfs = [
        meta_df.iloc[0 : cfg.data.n_train],
        meta_df.iloc[cfg.data.n_train : cfg.data.n_train + cfg.data.n_val],
        meta_df.iloc[cfg.data.n_train + cfg.data.n_val : cfg.data.n_train + cfg.data.n_val + cfg.data.n_test],
    ]

    data_module = Poisson1DDataModule(
        root_dir=cfg.data.root_dir,
        batch_size=cfg.data.batch_size,
        meta_dfs=meta_dfs,
        normalize_type=cfg.data.normalize_type,
        rtol=cfg.data.rtol,
    )

    ## SETUP WANDB LOGGER
    logger = WandbLogger(**cfg.train.logger, settings=wandb.Settings(start_method="thread"))
    logger.watch(gbms)

    # CALLBACKS SETUP
    callbacks = [
        EarlyStopping(**cfg.train.early_stopping),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if cfg.train.save_model:
        callbacks.append(ModelCheckpoint(monitor=cfg.train.monitor, mode="min", verbose=True, save_last=True))

    # TRAINER
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    # TRAIN
    trainer.fit(model=gbms, datamodule=data_module)

    # TEST
    if cfg.train.test:
        trainer.test(ckpt_path="best", datamodule=data_module)


# %%
if __name__ == "__main__":
    main()

# %%
