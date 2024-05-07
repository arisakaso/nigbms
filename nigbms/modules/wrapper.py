import pydevd  # noqa
import torch
from torch import Tensor, randn_like  # noqa
from torch.autograd import Function, grad
from hydra.utils import instantiate
from nigbms.modules.solvers import _Solver
from nigbms.utils.convert import tensordict2list
from nigbms.utils.solver import bms
from nigbms.utils.solver import rademacher_like  # noqa


class register_custom_grad_fn(Function):
    @staticmethod
    def forward(ctx, d: dict, *thetas):
        ctx.d = d
        ctx.save_for_backward(*thetas)

        return d["y"].detach()

    @staticmethod
    def backward(ctx, grad_y):
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)  # for debugging
        y = ctx.d["y"]
        theta = ctx.d["theta"].clone().detach()
        theta.apply(lambda x: x.requires_grad_())
        _, thetas = tensordict2list(theta)
        wrapper = ctx.d["wrapper"]
        cfg = wrapper.cfg

        def f(x):
            return wrapper.solver(ctx.d["tau"], x)

        def f_hat(x):
            return wrapper.surrogate(ctx.d["tau"], x)

        f_fwds = [None] * cfg.Nv
        f_hat_trues = [None] * cfg.Nv
        cv_fwds = [None] * cfg.Nv

        for i in range(cfg.Nv):
            v = rademacher_like(theta)
            _, vs = tensordict2list(v)

            with torch.no_grad():
                if cfg.jvp_type == "forwardAD":
                    _, dvf = torch.func.jvp(f, (theta,), (v,))
                elif cfg.jvp_type == "forwardFD":
                    dvf = (f(theta + v * cfg.eps) - y) / cfg.eps
            dvL = torch.sum(grad_y * dvf, dim=1)
            f_fwds[i] = list(map(lambda x: bms(x, dvL), vs))

            if cfg.grad_type in ["f_hat_true", "cv_fwd"]:
                wrapper.opt.zero_grad()
                y_hat, dvf_hat = torch.func.jvp(f_hat, (theta,), (v,))  # forward AD
                f_hat_trues[i] = grad(y_hat, thetas, grad_outputs=grad_y, retain_graph=True)
                wrapper.loss(y, y_hat, dvf, dvf_hat)["s_loss"].mean().backward(
                    inputs=list(wrapper.surrogate.parameters())
                )
                if wrapper.clip:
                    torch.nn.utils.clip_grad_norm_(wrapper.surrogate.parameters(), wrapper.clip)

                wrapper.opt.step()

                dvL_hat = torch.sum(grad_y * dvf_hat, dim=1)
                f_hat_fwd = map(lambda x: bms(x, dvL_hat), vs)
                cv_fwds[i] = list(map(lambda x, y, z: x - (y - z), f_fwds[i], f_hat_fwd, f_hat_trues[i]))

        grad_thetas = [torch.stack(x).mean(dim=0) for x in zip(*eval(cfg.grad_type + "s"), strict=False)]

        return None, *grad_thetas


# TODO: refactor this using dataclass
class WrappedSolver(_Solver):
    def __init__(self, solver: _Solver, surrogate: _Solver, opt, loss, clip, cfg: dict) -> None:
        super().__init__(solver.params_fix, surrogate.params_learn)
        self.solver = solver
        self.surrogate = surrogate
        self.opt = instantiate(opt, params=self.surrogate.parameters())
        self.loss = instantiate(loss)
        self.clip = clip
        self.cfg = cfg

    def forward(self, tau: dict, theta: Tensor) -> Tensor:
        y = self.solver(tau, theta)
        if self.cfg.grad_type == "f_true":
            return y
        else:
            _, thetas = tensordict2list(theta)
            d = {
                "tau": tau,
                "theta": theta,
                "y": y,
                "wrapper": self,
            }
            y = register_custom_grad_fn.apply(d, *thetas)
            return y
