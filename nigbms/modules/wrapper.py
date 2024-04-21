import torch
from torch import Tensor
from torch.autograd import grad

from nigbms.modules.solvers import _Solver
from nigbms.utils.solver import add_tensordicts, rademacher_like


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
        if isinstance(x, Tensor):
            x_plus = x + eps * v
        else:
            x_plus = add_tensordicts(x, v.apply(lambda x: eps * x))

        y = f(x)
        y_plus = f(x_plus)
        dvf = (y_plus - y) / eps

    elif jvp_type == "centralFD":
        if isinstance(x, Tensor):
            x_plus = x + eps * v
            x_minus = x - eps * v
        else:
            x_plus = add_tensordicts(x, v.apply(lambda x: eps * x))
            x_minus = add_tensordicts(x, v.apply(lambda x: -eps * x))

        y = f(x)
        dvf = (f(x_plus) - f(x_minus)) / (2 * eps)

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
        v = v * d["v_scale"]
        bs, in_dim = x.shape

        if d["grad_type"] == "f_true":
            # full gradient of f
            try:
                f_true = grad(d["y"], x, grad_outputs=grad_y, retain_graph=True)[0]
            except:
                f_true = torch.zeros_like(x)

            return f_true, None

        # forward gradient of f
        f_fwd = torch.sum(grad_y * d["dvf"], dim=1, keepdim=True) * v
        f_fwd = f_fwd.reshape(d["Nv"], bs, in_dim).mean(dim=0)  # average over Nv

        if d["grad_type"] == "f_fwd":
            return f_fwd, None

        # forward gradient of f_hat
        f_hat_fwd = torch.sum(grad_y * d["dvf_hat"], dim=1, keepdim=True) * v
        f_hat_fwd = f_hat_fwd.reshape(d["Nv"], bs, in_dim).mean(dim=0)  # average over Nv

        # full gradient of f_hat
        f_hat_true = grad(d["y_hat"], x, grad_outputs=grad_y, retain_graph=True)[0] / d["Nv"]

        if d["grad_type"] == "f_hat_true":
            return f_hat_true, None

        elif d["grad_type"] == "cv_fwd":
            # control variates
            # c = f_fwd.norm(dim=1, keepdim=True) / (f_hat_fwd - f_hat_true).norm(dim=1, keepdim=True)
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
            "y": y,
            "dvf": dvf,
            "y_hat": y_hat,
            "dvf_hat": dvf_hat,
            "grad_type": self.cfg["grad_type"],
        }
        y = register_custom_grad_fn.apply(theta, d)
        return y, y_hat, dvf, dvf_hat
