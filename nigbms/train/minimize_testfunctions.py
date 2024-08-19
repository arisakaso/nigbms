import hydra
import lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from nigbms.configs.modules.meta_solvers.configs import ConstantMetaSolverConfig  # noqa
from nigbms.configs.modules.solvers.configs import TestFunctionConfig  # noqa
from nigbms.configs.modules.surrogates.configs import TestFunctionSurrogateConfig  # noqa
from nigbms.modules.tasks import MinimizeTestFunctionTask
from nigbms.modules.wrapper import WrappedSolver


#### TEST FUNCTIONS ####
def sphere(x) -> torch.Tensor:
    """Sphere function."""
    return torch.sum(x**2, dim=-1, keepdim=True)


def rosenbrock(x) -> torch.Tensor:
    """Rosenbrock function."""
    x1 = x[..., :-1]
    x2 = x[..., 1:]
    return torch.sum(100 * (x2 - x1**2) ** 2 + (1 - x1) ** 2, dim=-1, keepdim=True)


def rosenbrock_separate(x) -> torch.Tensor:
    """Rosenbrock function (easy)."""
    assert x.shape[-1] % 2 == 0, "Dimension must be even."
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.sum((1 - x1) ** 2 + 100 * (x2 - x1**2) ** 2, dim=-1, keepdim=True)


def rastrigin(x) -> torch.Tensor:
    """Rastrigin function."""
    A = 10
    n = x.shape[-1]
    return A * n + torch.sum(x**2 - A * torch.cos(x * torch.pi * 2), dim=-1, keepdim=True)


### MAIN ###
@hydra.main(version_base="1.3", config_path="../configs/train", config_name="minimize_testfunctions")
def main(cfg):
    # set up
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.wandb.project, config=wandb.config, mode=cfg.wandb.mode)
    torch.set_default_dtype(eval(cfg.dtype))
    torch.set_default_device(cfg.device)
    pl.seed_everything(seed=cfg.seed, workers=True)

    test_func = eval(cfg.problem.test_function)
    tau = MinimizeTestFunctionTask(None, test_func)
    meta_solver = instantiate(cfg.meta_solver)
    solver = instantiate(cfg.solver)
    surrogate = instantiate(cfg.surrogate)
    constructor = instantiate(cfg.constructor)
    loss = torch.sum
    opt = instantiate(cfg.opt, params=list(meta_solver.parameters()))
    wrapped_solver = WrappedSolver(solver=solver, surrogate=surrogate, constructor=constructor, **cfg.wrapper)

    for i in range(1, cfg.problem.num_iter + 1):
        # clear gradients
        opt.zero_grad()

        # forward pass
        theta = meta_solver(tau)
        y = wrapped_solver(tau, theta)

        # backprop for theta
        loss(y).backward(inputs=theta, create_graph=True)

        # logging
        ref = theta.clone()  # copy to get the true gradient
        f_true = torch.autograd.grad(test_func(ref).sum(), ref)[0]
        sim = torch.cosine_similarity(f_true, theta.grad, dim=1, eps=1e-20).detach()
        wandb.log({"ymean": y.mean(), "ymax": y.max(), "ymin": y.min(), "cos_sim": sim.mean()})
        if i % 100 == 0:
            print(f"{i}, ymean: {y.mean():.3g}, ymax: {y.max():.3g}, ymin: {y.min():.3g}, cos_sim: {sim.mean():.3g}")

        # clip gradients
        if cfg.clip:
            torch.nn.utils.clip_grad_norm_(meta_solver.parameters(), cfg.clip)

        # update
        opt.step()

    return y.mean()


if __name__ == "__main__":
    main()
