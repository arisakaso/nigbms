# %%
import hydra
import torch
import wandb
from hydra.utils import instantiate
from lightning import LightningModule, Trainer, seed_everything
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers.wandb import WandbLogger

from nigbms.modules.wrapper import WrappedSolver
from nigbms.utils.resolver import calc_in_channels, calc_in_dim

OmegaConf.register_new_resolver("calc_in_dim", calc_in_dim)
OmegaConf.register_new_resolver("calc_in_channels", calc_in_channels)
OmegaConf.register_new_resolver("eval", eval)


# %%
class NIGBMS(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False

        self.cfg = cfg
        self.meta_solver = instantiate(cfg.meta_solver)
        self.solver = instantiate(cfg.solver)
        self.surrogate = instantiate(cfg.surrogate)
        self.wrapped_solver = WrappedSolver(solver=self.solver, surrogate=self.surrogate, **cfg.wrapper)
        self.loss = instantiate(cfg.loss)
        if cfg.compile:
            self.solver.compile()
            self.wrapped_solver.compile()
            self.surrogate.compile()

    def on_fit_start(self):
        seed_everything(seed=self.cfg.seed, workers=True)

    def _add_prefix(self, d, prefix):
        return dict([(prefix + k, v) for k, v in d.items()])

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        tau = batch
        theta = self.meta_solver(tau)
        if self.cfg.logging:
            theta["x0"].retain_grad()
        y = self.wrapped_solver(tau, theta)
        tau.features["xn"] = self.wrapped_solver.solver.x  # add xn for surrogate input
        loss_dict = self.loss(tau, theta, y)
        self.manual_backward(loss_dict["loss"], create_graph=True, inputs=list(self.meta_solver.parameters()))

        # logging
        if self.cfg.logging:
            theta_ref = theta.clone()  # copy to get the true gradient
            y_ref = self.solver(tau, theta_ref)
            loss_ref = self.loss(tau, theta_ref, y_ref)["loss"]
            f_true = torch.autograd.grad(loss_ref, theta_ref["x0"])[0]
            sim = torch.cosine_similarity(f_true, theta["x0"].grad, dim=1, eps=1e-20)
            self.log("train/sim", sim.mean(), prog_bar=True)

        opt.step()
        sch.step()

        self.log_dict(self._add_prefix(loss_dict, "train/"))
        if self.wrapped_solver.loss_dict:
            self.log_dict(self._add_prefix(self.wrapped_solver.loss_dict, "surrogate/"), prog_bar=True)

    def validation_step(self, batch, batch_idx):
        tau = batch
        theta = self.meta_solver(tau)
        y = self.solver(tau, theta)  # no surrogate
        loss_dict = self.loss(tau, theta, y)

        self.log_dict(self._add_prefix(loss_dict, "val/"))

    def test_step(self, batch, batch_idx):
        tau = batch
        theta = self.meta_solver(tau)
        y = self.solver(tau, theta)  # no surrogate
        loss_dict = self.loss(tau, theta, y)

        self.log_dict(self._add_prefix(loss_dict, "test/"))

    def configure_optimizers(self):
        opt = instantiate(self.cfg.opt, params=self.meta_solver.parameters())
        sch = instantiate(self.cfg.sch, optimizer=opt)

        return {
            "optimizer": opt,
            "lr_scheduler": sch,
            "monitor": self.cfg.monitor,
        }


@hydra.main(version_base="1.3", config_path="../configs/train", config_name="poisson1d")
def main(cfg: DictConfig):
    seed_everything(seed=cfg.seed, workers=True)
    torch.set_default_dtype(eval(cfg.dtype))
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.wandb.project, config=wandb.config, mode=cfg.wandb.mode)
    logger = WandbLogger(settings=wandb.Settings(start_method="thread"))

    callbacks = [instantiate(c) for c in cfg.callbacks]
    data_module = instantiate(cfg.data)

    nigbms = NIGBMS(cfg)

    trainer = Trainer(logger=logger, callbacks=callbacks, **cfg.trainer)
    trainer.fit(model=nigbms, datamodule=data_module)

    # TEST
    if cfg.test:
        trainer.test(ckpt_path="best", datamodule=data_module)


# %%
if __name__ == "__main__":
    main()

# %%
