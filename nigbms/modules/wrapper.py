import pydevd  # noqa
import torch
from torch import Tensor, randn_like  # noqa
from torch.autograd import Function, grad
from hydra.utils import instantiate
from nigbms.modules.solvers import _Solver
from nigbms.modules.data import PyTorchLinearSystemTask
from nigbms.utils.solver import rademacher_like
from nigbms.modules.constructors import ThetaConstructor  # noqa
from omegaconf import DictConfig


class register_custom_grad(Function):
    """Autograd function for custom gradient computation.
    During backward, the surrogate model in wrapper is trained.
    """

    @staticmethod
    def forward(ctx, wrapper, theta):
        ctx.theta = theta
        ctx.wrapper = wrapper
        return wrapper.y

    @staticmethod
    def backward(ctx, grad_y):
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)  # for debugging
        theta = ctx.theta.detach().requires_grad_()
        wrapper = ctx.wrapper
        hparams = ctx.wrapper.hparams

        f_fwd = torch.zeros(hparams.Nv, *theta.shape, device=theta.device)
        f_hat_true = torch.zeros(hparams.Nv, *theta.shape, device=theta.device)
        cv_fwd = torch.zeros(hparams.Nv, *theta.shape, device=theta.device)

        for i in range(hparams.Nv):
            with torch.no_grad():
                # sample random vector
                if hparams.v_dist == "rademacher":
                    v = rademacher_like(theta)
                elif hparams.v_dist == "normal":
                    v = randn_like(theta)
                else:
                    raise ValueError("v_dist must be 'rademacher' or 'normal'")

                # compute forward gradients
                if hparams.jvp_type == "forwardAD":
                    _, dvf = torch.func.jvp(wrapper.f, (theta,), (v,))
                elif hparams.jvp_type == "forwardFD":
                    dvf = (wrapper.f(theta + v * hparams.eps) - wrapper.y) / hparams.eps
                dvL = torch.sum(grad_y * dvf, dim=1, keepdim=True)
                f_fwd[i] = dvL * v

            if hparams.grad_type in ["f_hat_true", "cv_fwd"]:
                wrapper.opt.zero_grad()

                # compute control forward gradient
                y_hat, dvf_hat = torch.func.jvp(wrapper.f_hat, (theta,), (v,))  # forward AD
                f_hat_true[i] = grad(y_hat, theta, grad_outputs=grad_y, retain_graph=True)[0]
                dvL_hat = torch.sum(grad_y * dvf_hat, dim=1, keepdim=True)
                f_hat_fwd = dvL_hat * v
                cv_fwd[i] = f_fwd[i] - f_hat_fwd + f_hat_true[i]

                # training surrogate
                wrapper.loss_dict = wrapper.loss(wrapper.y, y_hat, dvf, dvf_hat, dvL, dvL_hat)
                wrapper.loss_dict["loss"].backward(inputs=list(wrapper.surrogate.parameters()))
                if hparams.clip:
                    torch.nn.utils.clip_grad_norm_(wrapper.surrogate.parameters(), hparams.clip)
                wrapper.opt.step()

        with torch.no_grad():
            grad_theta = torch.mean(eval(hparams.grad_type), dim=0)

        return None, grad_theta


class WrappedSolver(_Solver):
    """Wrapper class for the solver and surrogate solver."""

    def __init__(
        self, solver: _Solver, surrogate: _Solver, constructor: ThetaConstructor, hparams: DictConfig
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
