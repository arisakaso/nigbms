import pydevd  # noqa
import torch
from torch import Tensor
from torch.autograd import Function, grad

from nigbms.modules.solvers import _Solver
from nigbms.utils.convert import tensordict2list
from nigbms.utils.solver import bms, rademacher_like


def jvp(f, x: Tensor, v: Tensor, jvp_type: str, eps: float):
    assert jvp_type in ["forwardAD", "forwardFD", "centralFD"]
    assert type(x) == type(v)
    assert x.shape == v.shape

    if jvp_type == "forwardAD":
        y, dvf = torch.func.jvp(f, (x,), (v,))

    elif jvp_type == "forwardFD":
        y = f(x)
        dvf = (f(x + v * eps) - y) / eps

    elif jvp_type == "centralFD":
        y = f(x)
        dvf = (f(x + v * eps) - f(x - v * eps)) / (2 * eps)

    return y, dvf


class register_custom_grad_fn(Function):
    @staticmethod
    def forward(ctx, d: dict, *thetas):
        """_summary_

        Args:
            x: input
            d: dictionary with keys: y, dvf, y_hat, dvf_hat, grad_type

        Returns:
            y
        """
        ctx.d = d
        ctx.save_for_backward(*thetas)

        return d["y"].detach()

    @staticmethod
    def backward(ctx, grad_y):
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)  # for debugging
        d = ctx.d
        theta = ctx.d["theta"].clone().detach()
        theta.apply(lambda x: x.requires_grad_())
        _, thetas = tensordict2list(theta)

        f_fwds = [None] * d["cfg"].Nv
        f_hat_trues = [None] * d["cfg"].Nv
        cv_fwds = [None] * d["cfg"].Nv

        for i in range(d["cfg"].Nv):
            v = rademacher_like(theta)
            _, vs = tensordict2list(v)

            with torch.no_grad():
                _, dvf = torch.func.jvp(d["f"], (theta,), (v,))  # forward AD
            dvL = torch.sum(grad_y * dvf, dim=1)
            f_fwds[i] = list(map(lambda x: bms(x, dvL), vs))

            if d["cfg"].grad_type in ["f_hat_true", "cv_fwd"]:
                d["s_opt"].zero_grad()
                y_hat, dvf_hat = torch.func.jvp(d["f_hat"], (theta,), (v,))  # forward AD
                f_hat_trues[i] = grad(y_hat, thetas, grad_outputs=grad_y, retain_graph=True)
                d["s_loss"](d["y"], y_hat, dvf, dvf_hat)["s_loss"].mean().backward()
                d["s_opt"].step()

                dvL_hat = torch.sum(grad_y * dvf_hat, dim=1)
                f_hat_fwd = map(lambda x: bms(x, dvL_hat), vs)
                cv_fwds[i] = list(map(lambda x, y, z: x - (y - z), f_fwds[i], f_hat_fwd, f_hat_trues[i]))

        grad_thetas = [torch.stack(x).mean(dim=0) for x in zip(*eval(d["cfg"].grad_type + "s"), strict=False)]

        return None, *grad_thetas


class WrappedSolver(_Solver):
    def __init__(self, solver: _Solver, surrogate: _Solver, s_opt, s_loss, cfg: dict) -> None:
        super().__init__(solver.params_fix, surrogate.params_learn)
        self.solver = solver
        self.surrogate = surrogate
        self.s_opt = s_opt
        self.s_loss = s_loss
        self.cfg = cfg

    def forward(self, tau: dict, theta: Tensor) -> Tensor:
        y = self.solver(tau, theta)
        if self.cfg.grad_type == "f_true":
            return y
        else:
            _, thetas = tensordict2list(theta)
            d = {
                "y": y,
                "theta": theta,
                "f": lambda x: self.solver(tau, x),
                "f_hat": lambda x: self.surrogate(tau, x),
                "s_opt": self.s_opt,
                "s_loss": self.s_loss,
                "cfg": self.cfg,
            }
            y = register_custom_grad_fn.apply(d, *thetas)
            return y
