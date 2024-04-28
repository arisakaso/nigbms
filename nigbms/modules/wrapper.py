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
    def forward(ctx, d: dict, keys: list, *thetas):
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
        thetas = ctx.saved_tensors
        _, vs = tensordict2list(d["v"])

        if d["grad_type"] == "f_true":
            f_true = grad(d["y"], thetas, grad_outputs=grad_y, retain_graph=True)
            return None, None, *f_true

        dvL = torch.sum(grad_y * d["dvf"], dim=1)
        f_fwd = map(lambda x: bms(x, dvL), vs)
        if d["grad_type"] == "f_fwd":
            return None, None, *f_fwd

        f_hat_true = grad(d["y_hat"], thetas, grad_outputs=grad_y, retain_graph=True)
        if d["grad_type"] == "f_hat_true":
            return None, None, *f_hat_true

        dvL_hat = torch.sum(grad_y * d["dvf_hat"], dim=1)
        f_hat_fwd = map(lambda x: bms(x, dvL_hat), vs)
        cv_fwd = map(lambda x, y, z: x - (y - z), f_fwd, f_hat_fwd, f_hat_true)
        return None, None, *cv_fwd


class WrappedSolver(_Solver):
    def __init__(
        self,
        solver: _Solver,
        surrogate: _Solver,
        cfg: dict,
    ) -> None:
        super().__init__(solver.params_fix, surrogate.params_learn)
        self.solver = solver
        self.surrogate = surrogate
        self.cfg = cfg

    def _setup(self, tau: dict):
        # fix tau
        self._f = lambda x: self.solver(tau, x)
        self._f_hat = lambda x: self.surrogate(tau, x)

    def forward(self, tau: dict, theta: Tensor) -> Tensor:
        self._setup(tau)
        v = rademacher_like(theta)
        y, dvf = jvp(self._f, theta, v, self.cfg["jvp_type"], self.cfg["eps"])
        y_hat, dvf_hat = jvp(self._f_hat, theta, v, "forwardAD", 0.0)
        d = {
            "v": v,
            "y": y.clone(),
            "dvf": dvf.clone(),
            "y_hat": y_hat.clone(),
            "dvf_hat": dvf_hat.clone(),
            "grad_type": self.cfg["grad_type"],
        }
        ks, thetas = tensordict2list(theta)
        y = register_custom_grad_fn.apply(d, ks, *thetas)
        return y, y_hat, dvf, dvf_hat
