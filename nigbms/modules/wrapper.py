import torch
from torch import Tensor
from torch.autograd import grad

from nigbms.modules.solvers import _Solver
from nigbms.utils.solver import rademacher_like


def jvp(f, x: Tensor, v: Tensor, jvp_type: str, eps: float):
    assert jvp_type in ["forwardAD", "forwardFD", "centralFD"]
    assert type(x) == type(v)
    assert x.shape == v.shape

    if jvp_type == "forwardAD":
        y, dvf = torch.func.jvp(f, (x,), (v,))
        # with fwAD.dual_level():
        #     dual_input = fwAD.make_dual(x, v)
        #     dual_output = f(dual_input)
        #     y, dvf = fwAD.unpack_dual(dual_output)

    elif jvp_type == "forwardFD":
        y = f(x)
        dvf = (f(x + v * eps) - y) / eps

    elif jvp_type == "centralFD":
        y = f(x)
        dvf = (f(x + v * eps) - f(x - v * eps)) / (2 * eps)

    return y, dvf


class register_custom_grad_fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, d: dict):
        """_summary_

        Args:
            x: input
            d: dictionary with keys: y, dvf, y_hat, dvf_hat, grad_type

        Returns:
            y
        """
        ctx.d = d
        ctx.save_for_backward(x, d["v"])

        return d["y"].detach()

    @staticmethod
    def backward(ctx, grad_y):
        x, v = ctx.saved_tensors
        d = ctx.d

        if d["grad_type"] == "f_true":
            f_true = grad(d["y"], x, grad_outputs=grad_y, retain_graph=True)[0]
            return f_true, None

        # forward gradient of f
        f_fwd = torch.sum(grad_y * d["dvf"], dim=1, keepdim=True) * v
        if d["grad_type"] == "f_fwd":
            return f_fwd, None

        # forward gradient of f_hat
        f_hat_fwd = torch.sum(grad_y * d["dvf_hat"], dim=1, keepdim=True) * v

        # full gradient of f_hat
        f_hat_true = grad(d["y_hat"], x, grad_outputs=grad_y, retain_graph=True)[0] / d["Nv"]

        if d["grad_type"] == "f_hat_true":
            return f_hat_true, None

        elif d["grad_type"] == "cv_fwd":
            # control variates
            cv_fwd = f_fwd - (f_hat_fwd - f_hat_true)
            return cv_fwd, None

        else:
            raise NotImplementedError


class WrappedSolver(_Solver):
    def __init__(
        self,
        params_fix: dict,
        params_learn: dict,
        solver: _Solver,
        surrogate: _Solver,
        cfg: dict,
    ) -> None:
        super().__init__(params_fix, params_learn)
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
            "y": y,
            "dvf": dvf,
            "y_hat": y_hat,
            "dvf_hat": dvf_hat,
            "grad_type": self.cfg["grad_type"],
        }
        y = register_custom_grad_fn.apply(theta, d)
        return y, y_hat, dvf, dvf_hat
