import pydevd  # noqa
import torch
from torch import Tensor, randn_like  # noqa
from torch.autograd import Function, grad
from hydra.utils import instantiate
from nigbms.modules.solvers import _Solver
from nigbms.data.data_modules import Task
from nigbms.utils.solver import rademacher_like  # noqa


class register_custom_grad(Function):
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
        cfg = ctx.wrapper.cfg

        f_fwd = torch.zeros(cfg.Nv, *theta.shape, device=theta.device)
        f_hat_true = torch.zeros(cfg.Nv, *theta.shape, device=theta.device)
        cv_fwd = torch.zeros(cfg.Nv, *theta.shape, device=theta.device)

        for i in range(cfg.Nv):
            with torch.no_grad():
                v = rademacher_like(theta)

                if cfg.jvp_type == "forwardAD":
                    _, dvf = torch.func.jvp(wrapper.f, (theta,), (v,))
                elif cfg.jvp_type == "forwardFD":
                    dvf = (wrapper.f(theta + v * cfg.eps) - wrapper.y) / cfg.eps
                dvL = torch.sum(grad_y * dvf, dim=1, keepdim=True)
                f_fwd[i] = dvL * v

            if cfg.grad_type in ["f_hat_true", "cv_fwd"]:
                wrapper.opt.zero_grad()
                y_hat, dvf_hat = torch.func.jvp(wrapper.f_hat, (theta,), (v,))  # forward AD
                f_hat_true[i] = grad(y_hat, theta, grad_outputs=grad_y, retain_graph=True)[0]
                dvL_hat = torch.sum(grad_y * dvf_hat, dim=1, keepdim=True)
                f_hat_fwd = dvL_hat * v
                cv_fwd[i] = f_fwd[i] - f_hat_fwd + f_hat_true[i]

                wrapper.loss_dict = wrapper.loss(wrapper.y, y_hat, dvf, dvf_hat, dvL, dvL_hat)
                wrapper.loss_dict["loss"].backward(inputs=list(wrapper.surrogate.parameters()))
                if wrapper.clip:
                    torch.nn.utils.clip_grad_norm_(wrapper.surrogate.parameters(), wrapper.clip)
                wrapper.opt.step()

        with torch.no_grad():
            grad_theta = torch.mean(eval(cfg.grad_type), dim=0)

        return None, grad_theta


class WrappedSolver(_Solver):
    def __init__(self, solver: _Solver, surrogate: _Solver, constructor, opt, loss, clip, cfg: dict) -> None:
        super().__init__(solver.params_fix, surrogate.params_learn)
        self.solver = solver
        self.surrogate = surrogate
        self.constructor = constructor
        self.opt = instantiate(opt, params=self.surrogate.parameters())
        self.loss = instantiate(loss)
        self.clip = clip
        self.cfg = cfg
        self.loss_dict = None
        self.y = None

    def forward(self, tau: Task, theta: Tensor, mode: str = "train") -> Tensor:
        def f(x):
            x = self.constructor(x)
            y = self.solver(tau, x)
            return y

        def f_hat(x):
            x = self.constructor(x)
            y = self.surrogate(tau, x)
            return y

        self.f = f
        self.f_hat = f_hat

        y = self.f(theta)

        if mode == "test" or self.cfg.grad_type == "f_true":
            return y
        else:
            self.y = y.detach()
            return register_custom_grad.apply(self, theta)
