import collections
import logging

import hydra
import torch
import wandb
from hydra.utils import instantiate
from lightning import LightningModule, Trainer, seed_everything
from nigbms.configs import data, meta_solvers, solvers, surrogates, wrapper  # noqa
from nigbms.solvers import _PytorchIterativeSolver
from nigbms.utils.args import arrange_sweep_args_for_hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger

log = logging.getLogger(__name__)


class NIGBMS(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False

        self.cfg = cfg
        self.meta_solver = instantiate(cfg.meta_solver)
        self.solver = instantiate(cfg.solver)
        self.constructor = instantiate(cfg.constructor)
        self.surrogate = instantiate(cfg.surrogate, constructor=self.constructor)
        self.wrapped_solver = instantiate(
            cfg.wrapper, solver=self.solver, surrogate=self.surrogate, constructor=self.constructor
        )
        self.loss = instantiate(cfg.loss, constructor=self.constructor)
        if cfg.compile:  # This doesn't speed up the training, even slower. TODO: Investigate why?
            self.meta_solver = torch.compile(self.meta_solver)
            self.solver = torch.compile(self.solver)
            self.surrogate = torch.compile(self.surrogate)

        ref_solver_cfg = cfg.solver.copy()
        ref_solver_cfg.params_learn = {}
        self.ref_solver = instantiate(ref_solver_cfg)

    def on_fit_start(self):
        seed_everything(seed=self.cfg.seed, workers=True)

    def _add_prefix(self, d, prefix):
        return dict([(prefix + k, v) for k, v in d.items()])

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        tau = batch
        theta = self.meta_solver(tau)
        theta.retain_grad()  # logging purpose
        y = self.wrapped_solver(tau, theta)

        loss_dict = self.loss(tau, theta, y)
        self.manual_backward(loss_dict["loss"])

        # log the cosine similarity between the true gradient and the surrogate gradient
        # TODO: make this callback(?)
        if self.cfg.logging and self.cfg.wrapper.hparams.grad_type != "f_true":
            assert isinstance(self.solver, _PytorchIterativeSolver), "Only PytorchIterativeSolver is supported"
            theta_ref = theta.clone()  # copy to get the true gradient
            y_ref = self.wrapped_solver(tau, theta_ref, mode="test")
            loss_ref = self.loss(tau, theta_ref, y_ref)["loss"]
            f_true = torch.autograd.grad(loss_ref, theta_ref)[0]
            sim = torch.cosine_similarity(f_true, theta.grad, dim=1, eps=1e-20)
            self.log(
                "surrogate/sim",
                sim.mean(),
                logger=True,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                batch_size=tau.batch_size[0],
            )

        if self.current_epoch >= self.cfg.warmup:
            opt.step()

        self.log_dict(
            self._add_prefix(loss_dict, "train/"),
            logger=True,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=tau.batch_size[0],
        )
        if self.wrapped_solver.loss_dict:
            self.log_dict(
                self._add_prefix(self.wrapped_solver.loss_dict, "surrogate/"),
                logger=True,
                prog_bar=True,
                on_epoch=True,
                on_step=False,
                batch_size=tau.batch_size[0],
            )

    def on_train_epoch_end(self):
        if self.current_epoch >= self.cfg.warmup:
            self.lr_schedulers().step()
        if self.cfg.reset_opt:
            self.wrapped_solver.opt.state = collections.defaultdict(dict)  # Reset state

    def validation_step(self, batch, batch_idx):
        tau = batch
        theta = self.meta_solver(tau)
        y = self.wrapped_solver(tau, theta, mode="test")  # no surrogate
        loss_dict = self.loss(tau, theta, y)

        self.log_dict(
            self._add_prefix(loss_dict, "val/"),
            logger=True,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=tau.batch_size[0],
        )

    def test_step(self, batch, batch_idx):
        tau = batch

        theta = self.meta_solver(tau)
        y = self.wrapped_solver(tau, theta, mode="test")  # no surrogate
        loss_dict = self.loss(tau, theta, y)
        self.log_dict(
            self._add_prefix(loss_dict, "test/"),
            logger=True,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=tau.batch_size[0],
        )

        theta_dict = self.constructor(theta)
        y_ref = self.ref_solver(tau, theta_dict)
        loss_dict_ref = self.loss(tau, theta, y_ref)
        self.log_dict(
            self._add_prefix(loss_dict_ref, "ref/"),
            logger=True,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            batch_size=tau.batch_size[0],
        )

    def configure_optimizers(self):
        opt = instantiate(self.cfg.opt, params=self.meta_solver.model.parameters())
        sch = instantiate(self.cfg.sch, optimizer=opt)

        return {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": self.cfg.monitor,
        }


@hydra.main(version_base="1.3", config_path=".", config_name="train_small")
def main(cfg: DictConfig):
    # log.info(OmegaConf.to_yaml(cfg))
    seed_everything(seed=cfg.seed, workers=True)
    torch.set_default_dtype(eval(cfg.dtype))

    if cfg.wandb is not None:  # WandbLogger
        wandb.require("core")
        wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        logger = WandbLogger(
            settings=wandb.Settings(start_method="thread"), project=cfg.wandb.project, config=wandb.config
        )
    else:
        logger = None

    callbacks = [instantiate(c) for c in cfg.callbacks]
    data_module = instantiate(cfg.data)

    nigbms = NIGBMS(cfg)

    trainer = Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)
    trainer.fit(model=nigbms, datamodule=data_module)

    # TEST
    if cfg.test:
        trainer.test(ckpt_path="last", datamodule=data_module)

    wandb.finish()


# %%
if __name__ == "__main__":
    arrange_sweep_args_for_hydra()
    main()
