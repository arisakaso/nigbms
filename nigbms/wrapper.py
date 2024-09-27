from dataclasses import dataclass

import pydevd  # noqa
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor, randn_like  # noqa
from torch.autograd import Function, grad

from nigbms.constructors import ThetaConstructor  # noqa
from nigbms.data import PyTorchLinearSystemTask
from nigbms.solvers import AbstractSolver
from nigbms.utils.solver import rademacher_like


@dataclass
class NIGBMSConfig:
    y: Tensor
    f: callable
    f_hat: torch.nn.Module
    opt: torch.optim.Optimizer
    loss: torch.nn.Module
    v_dist: str
    grad_type: str
    jvp_type: str
    eps: float
    Nv: int
    v_scale: float
    clip: float
    additional_steps: int


class register_custom_grad(Function):
    """Autograd function for custom gradient computation.
    During backward, the surrogate model in wrapper is trained.
    """

    @staticmethod
    def forward(ctx, theta, cfg: NIGBMSConfig):
        ctx.theta = theta
        ctx.cfg = cfg
        return cfg.y

    @staticmethod
    def backward(ctx, grad_y):
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)  # for debugging
        theta = ctx.theta.detach().requires_grad_()
        cfg = ctx.cfg

        # placeholders
        f_fwd = torch.zeros(cfg.Nv, *theta.shape, device=theta.device)
        f_hat_true = torch.zeros(cfg.Nv, *theta.shape, device=theta.device)
        cv_fwd = torch.zeros(cfg.Nv, *theta.shape, device=theta.device)

        for i in range(cfg.Nv):
            with torch.no_grad():  # f is black-box
                # sample random vector
                if cfg.v_dist == "rademacher":
                    v = rademacher_like(theta)
                elif cfg.v_dist == "normal":
                    v = randn_like(theta)
                else:
                    raise ValueError("v_dist must be 'rademacher' or 'normal'")

                # compute forward gradient of the black-box solver (f)
                if cfg.jvp_type == "forwardAD":
                    _, dvf = torch.func.jvp(cfg.f, (theta,), (v,))
                elif cfg.jvp_type == "forwardFD":
                    dvf = (cfg.f(theta + v * cfg.eps) - cfg.y) / cfg.eps
                else:
                    raise ValueError("jvp_type must be 'forwardAD' or 'forwardFD")

                dvL = torch.sum(grad_y * dvf, dim=1, keepdim=True)
                f_fwd[i] = dvL * v

            if cfg.grad_type in ["f_hat_true", "cv_fwd"]:
                with torch.enable_grad():  # f_hat is differentiable
                    cfg.opt.zero_grad()

                    # compute control forward gradient
                    y_hat, dvf_hat = torch.func.jvp(cfg.f_hat, (theta,), (v,))  # forward AD
                    f_hat_true[i] = grad(y_hat, theta, grad_outputs=grad_y, retain_graph=True)[0]
                    dvL_hat = torch.sum(grad_y * dvf_hat, dim=1, keepdim=True)
                    f_hat_fwd = dvL_hat * v
                    cv_fwd[i] = cfg.v_scale * (f_fwd[i] - f_hat_fwd) + f_hat_true[i]

                    # training surrogate
                    parameters = [p for p in cfg.surrogate.parameters() if p.requires_grad]
                    cfg.loss_dict = cfg.loss(cfg.y, y_hat, dvf, dvf_hat, dvL, dvL_hat)
                    cfg.loss_dict["loss"].backward(inputs=parameters)
                    if isinstance(cfg.clip, float):
                        torch.nn.utils.clip_grad_norm_(cfg.surrogate.parameters(), cfg.clip)
                    cfg.opt.step()

                    for _ in range(cfg.additional_steps):
                        cfg.opt.zero_grad()
                        _y_hat, _dvf_hat = torch.func.jvp(cfg.f_hat, (theta,), (v,))  # forward AD
                        _dvL_hat = torch.sum(grad_y * _dvf_hat, dim=1, keepdim=True)
                        _loss_dict = cfg.loss(cfg.y, _y_hat, dvf, _dvf_hat, dvL, _dvL_hat)
                        _loss_dict["loss"].backward(inputs=parameters)
                        if isinstance(cfg.clip, float):
                            torch.nn.utils.clip_grad_norm_(cfg.surrogate.parameters(), cfg.clip)
                        cfg.opt.step()

        with torch.no_grad():
            grad_theta = torch.mean(eval(cfg.grad_type), dim=0)  # average over Nv

        return None, grad_theta


class WrappedSolver(AbstractSolver):
    """Wrapper class for the solver and surrogate solver."""

    def __init__(
        self, solver: AbstractSolver, surrogate: AbstractSolver, constructor: ThetaConstructor, hparams: DictConfig
    ) -> None:
        super().__init__(solver.params_fix, surrogate.params_learn)
        self.solver = solver
        self.surrogate = surrogate
        self.constructor = constructor
        self.opt = instantiate(hparams.opt, params=self.surrogate.parameters())
        self.loss = instantiate(hparams.loss)
        self.hparams = hparams
        self.loss_dict = None
        self.y = None

    def forward(self, tau: PyTorchLinearSystemTask, theta: Tensor, mode: str = "train") -> Tensor:
        def f(x: Tensor) -> Tensor:  # solve the problem
            x = self.constructor(x)
            y = self.solver(tau, x)
            return y  # y must be a tensor for the custom grad to work

        def f_hat(x: Tensor) -> Tensor:  # surrogate solver
            x = self.constructor(x)
            y = self.surrogate(tau, x)
            return y  # y must be a tensor for the custom grad to work

        self.f = f
        self.f_hat = f_hat

        y = self.f(theta)

        if mode == "test" or self.hparams.grad_type == "f_true":
            return y
        else:
            self.y = y.detach()  # y is computed by the black-box solver
            return register_custom_grad.apply(self, theta)
