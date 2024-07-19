# %%
import os
from typing import Dict

import hydra
import pytorch_lightning as pl
import torch
import torch_optimizer
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor
from torch.autograd import grad
from torch.nn.functional import cosine_similarity
from torch.nn.utils import clip_grad_norm_

from src.losses.pytorch import combine_losses
from src.models.surrogates import jvp, register_custom_grad_fn


def initialize_parameters_fcn(m, scale=1e-3):
    if isinstance(m, torch.nn.Linear):
        print("initialized with scale", m, scale)
        torch.nn.init.uniform_(m.weight, -scale, scale)
        torch.nn.init.uniform_(m.bias, -scale, scale)


class GBMSPoisson1DNonIntrusive(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(cfg, logger=False)

        self.meta_solver = instantiate(cfg.meta_solver)
        self.decoder = instantiate(cfg.decoder)
        self.solver = instantiate(cfg.solver)
        self.i_loss = instantiate(cfg.loss.i)
        self.d_loss = instantiate(cfg.loss.d)
        self.s_loss = instantiate(cfg.loss.s)
        self.surrogate = instantiate(cfg.surrogate)
        self.update_surrogate = cfg.wrapper.grad_type in ["cv_fwd", "f_hat_true"]

        if self.hparams.train.initialize_parameters_m:
            self.meta_solver.apply(initialize_parameters_fcn)
        if self.hparams.train.initialize_parameters_s:
            self.surrogate.apply(initialize_parameters_fcn)

        if self.solver.jittable:
            self.solver = torch.jit.script(self.solver)

    def on_fit_start(self):
        pl.seed_everything(seed=self.hparams.train.seed, workers=True)
        if self.hparams.train.save_data:
            self.save_dir = f"{self.hparams.data.root_dir}/{self.logger.experiment.id}"
            os.makedirs(self.save_dir)

    def construct_fs(self, tau: Dict, output_type: str):
        """define f, f_hat, and postprocessor for each tau

        Args:
            tau (Dict): _description_
        """
        bs, n, _ = tau["A"].shape

        if output_type == "loss":

            def _f(theta: Tensor):
                theta = self.decoder(theta)
                history = self.solver(tau, theta)  # (bs, nit, n, 1)
                y = self.d_loss(tau, history)["d_loss"]
                return y.reshape(bs, -1)

            def _f_hat(theta: Tensor):
                theta = self.decoder(theta)
                y_hat = self.surrogate(tau, theta)
                return y_hat.reshape(bs, -1)

            def _compute_losses(theta, y: Tensor):
                theta = self.decoder(theta)
                independent_losses = self.i_loss(tau, theta)
                dependent_losses = {"d_loss": y.ravel()}
                return combine_losses(independent_losses, dependent_losses)

        elif output_type == "history_err":

            def _f(theta: Tensor):
                theta = self.decoder(theta)
                y = self.solver(tau, theta)  # (bs, nit, n, 1)
                return y.reshape(bs, -1)

            def _f_hat(theta: Tensor):
                theta = self.decoder(theta)
                y_hat = self.surrogate(tau, theta)
                return y_hat.reshape(bs, -1)

            def _compute_losses(theta, y: Tensor):
                theta = self.decoder(theta)
                independent_losses = self.i_loss(tau, theta)
                dependent_losses = self.d_loss(tau, history_err=y)
                return combine_losses(independent_losses, dependent_losses)

        return _f, _f_hat, _compute_losses

    def compute_gradients(self, theta, v, y, y_hat, dvf, dvf_hat, y_grad):
        # y_grad = grad(loss, y, retain_graph=True)[0]
        cfg_wrapper = self.hparams.wrapper
        dvL = y_grad * dvf
        dvL_hat = y_grad * dvf_hat
        f_fwd = dvL.sum(dim=1, keepdim=True) * v * cfg_wrapper.v_scale
        f_hat_fwd = dvL_hat.sum(dim=1, keepdim=True) * v * cfg_wrapper.v_scale
        f_true = grad(y, theta, grad_outputs=y_grad, retain_graph=True)[0]
        f_hat_true = grad(y_hat, theta, grad_outputs=y_grad, retain_graph=True)[0]
        cv_fwd = f_fwd - (f_hat_fwd - f_hat_true)
        gradients = {
            "f_true": f_true,
            "f_fwd": f_fwd,
            "f_hat_true": f_hat_true,
            "f_hat_fwd": f_hat_fwd,
            "cv_fwd": cv_fwd,
        }
        return gradients

    def log_gradient_metrics(self, gradients):
        metrics = {
            "sim_f_fwd": cosine_similarity(
                gradients["f_fwd"], gradients["f_true"], dim=1
            ),
            "sim_cv_fwd": cosine_similarity(
                gradients["cv_fwd"], gradients["f_true"], dim=1
            ),
            "sim_f_hat_true": cosine_similarity(
                gradients["f_hat_true"], gradients["f_true"], dim=1
            ),
        }
        self.log_dict(
            self._add_prefix(metrics, "metrics/"),
            logger=True,
            on_epoch=True,
            on_step=False,
        )

    def _add_prefix(self, d, prefix):
        return dict([(prefix + k, v.mean()) for k, v in d.items()])

    def training_step(self, tau, batch_idx):
        cfg_wrap = self.hparams.wrapper  # shorthand
        cfg_train = self.hparams.train  # shorthand
        m_opt, s_opt = self.optimizers()
        theta = self.meta_solver(tau)
        f, f_hat, compute_losses = self.construct_fs(tau, cfg_wrap.output_type)

        # forward computation
        v = torch.randint(0, 2, size=theta.shape, device=theta.device) * 2.0 - 1
        y, dvf = jvp(f, theta, v, cfg_wrap.jvp_type, cfg_wrap.eps)
        y_hat, dvf_hat = jvp(f_hat, theta, v, cfg_wrap.jvp_type_s, cfg_wrap.eps)

        # register custom grad
        d = {
            "v": v,
            "y": y,
            "dvf": dvf,
            "y_hat": y_hat,
            "dvf_hat": dvf_hat,
            "grad_type": cfg_wrap.grad_type,
            "Nv": cfg_wrap.Nv,
            "v_scale": cfg_wrap.v_scale,
        }
        y_modified = register_custom_grad_fn.apply(theta, d)
        y_modified.retain_grad()

        # meta solver update
        m_opt.zero_grad()
        m_losses = compute_losses(theta, y_modified)
        m_loss = m_losses["m_loss"].mean()
        if self.current_epoch >= cfg_train.warmup_epochs:
            self.manual_backward(
                m_loss, retain_graph=True, inputs=list(self.meta_solver.parameters())
            )
            if cfg_train.clip.m:
                clip_grad_norm_(self.meta_solver.parameters(), cfg_train.clip.m)
            m_opt.step()
        self.log_dict(
            self._add_prefix(m_losses, "train/loss/"),
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

        # log gradients
        if cfg_train.log_gradients:
            gradients = self.compute_gradients(
                theta, v, y, y_hat, dvf, dvf_hat, y_modified.grad
            )
            self.log_gradient_metrics(gradients)

        # surrogate update
        s = cfg_wrap.update_steps  # shorthand
        for i in range(s):
            s_opt.zero_grad()
            s_losses = self.s_loss(
                y.clone().detach(), y_hat, dvf.clone().detach(), dvf_hat
            )
            s_loss = s_losses["s_loss"].mean()
            self.manual_backward(
                s_loss,
                retain_graph=True,
                inputs=list(self.surrogate.parameters()),
            )
            if cfg_train.clip.s:
                clip_grad_norm_(self.surrogate.parameters(), cfg_train.clip.s)
            s_opt.step()
            self.log_dict(
                self._add_prefix(s_losses, "train/loss/"),
                logger=True,
                on_epoch=True,
                on_step=False,
            )

    def on_train_epoch_end(self) -> None:
        m_sch, s_sch = self.lr_schedulers()
        m_sch.step()
        s_sch.step()

    def on_validation_epoch_start(self) -> None:
        pass

    def validation_step(self, tau, batch_idx, dataloader_idx):
        theta = self.meta_solver(tau)
        theta = self.decoder(theta)
        i_losses = self.i_loss(tau, theta)
        history = self.solver(tau, theta)
        d_losses = self.d_loss(tau, history)
        m_losses = combine_losses(i_losses, d_losses)
        self.log_dict(
            self._add_prefix(m_losses, "val/loss/"),
            logger=True,
            on_epoch=True,
            on_step=False,
        )

    def on_validation_epoch_end(self) -> None:
        pass

    def on_test_epoch_start(self) -> None:
        self.fixed_solver = instantiate(self.hparams.solver)
        self.fixed_solver.params_learn = {}

    def test_step(self, tau, batch_idx, dataloader_idx):
        theta = self.meta_solver(tau)
        theta = self.decoder(theta)
        i_losses = self.i_loss(tau, theta)
        history = self.solver(tau, theta)
        d_losses = self.d_loss(tau, history)
        m_losses = combine_losses(i_losses, d_losses)
        self.log_dict(
            self._add_prefix(m_losses, "test/loss/"),
            logger=True,
            on_epoch=True,
            on_step=False,
        )

        # non-learning baseline
        theta = torch.zeros_like(theta)
        history = self.fixed_solver(tau, theta)
        i_losses = self.i_loss(tau, theta)
        d_losses = self.d_loss(tau, history)
        m_losses = combine_losses(i_losses, d_losses)
        self.log_dict(
            self._add_prefix(m_losses, "baseline/loss/"),
            logger=True,
            on_epoch=True,
            on_step=False,
        )

    def configure_optimizers(self):
        cfg_opt = self.hparams.train.optimizer  # shorthand

        m_opt = instantiate(cfg_opt.m_opt, params=self.meta_solver.parameters())
        if cfg_opt.lookahead:
            m_opt = torch_optimizer.Lookahead(
                m_opt,
                k=cfg_opt.lookahead.k,
                alpha=cfg_opt.lookahead.alpha,
            )

        s_opt = instantiate(cfg_opt.s_opt, params=self.surrogate.parameters())
        if cfg_opt.s_lookahead:
            s_opt = torch_optimizer.Lookahead(
                s_opt,
                k=cfg_opt.s_lookahead.k,
                alpha=cfg_opt.s_lookahead.alpha,
            )

        m_sch = instantiate(cfg_opt.m_sch, optimizer=m_opt)
        s_sch = instantiate(cfg_opt.s_sch, optimizer=s_opt)

        return [m_opt, s_opt], [m_sch, s_sch]

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(
                        f"Skip loading parameter: {k}, "
                        f"required shape: {model_state_dict[k].shape}, "
                        f"loaded shape: {state_dict[k].shape}"
                    )
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)


def load_pretrained_model(model: pl.LightningModule, pretrained_model: str, cfg):
    api = wandb.Api()
    artifact = api.artifact(pretrained_model, type="model")
    artifact_dir = artifact.download(
        root="/workspace/gbms/data/results/pretrained_models"
    )
    model = model.load_from_checkpoint(
        artifact_dir + "/model.ckpt", strict=False, **cfg
    )
    return model


@hydra.main(
    config_path="../configs",
    config_name="train_beam",
    version_base=None,
)
def main(cfg: DictConfig):
    # FIX SEED
    pl.seed_everything(seed=cfg.train.seed, workers=True)

    # SET DEFAULT DTYPE
    torch.set_default_dtype(eval(cfg.common.dtype))

    # MODEL SETUP
    if cfg.train.pretrained_model:
        gbms = load_pretrained_model(
            GBMSPoisson1DNonIntrusive, cfg.train.pretrained_model, cfg
        )
    else:
        gbms = GBMSPoisson1DNonIntrusive(cfg)

    # DATA SETUP
    data_module = instantiate(cfg.data)

    # LOGGER SETUP
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        project=cfg.train.logger.project,
        config=wandb.config,
        mode=cfg.train.logger.mode,
    )
    logger = WandbLogger(
        **cfg.train.logger, settings=wandb.Settings(start_method="thread")
    )
    # logger.watch(gbms)

    # CALLBACKS SETUP
    callbacks = [
        EarlyStopping(**cfg.train.callbacks.early_stopping),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(**cfg.train.callbacks.checkpoint),
        ModelSummary(3),
    ]

    # TRAINER SETUP
    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        # profiler=AdvancedProfiler(filename="profile.txt"),
        **cfg.train.trainer,
    )

    # TRAIN
    trainer.validate(model=gbms, datamodule=data_module)
    trainer.fit(model=gbms, datamodule=data_module)

    # TEST
    if cfg.train.test:
        trainer.test(ckpt_path="best", datamodule=data_module)


# %%
if __name__ == "__main__":
    main()
