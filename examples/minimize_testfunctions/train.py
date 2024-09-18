# TODO: This example should be a minimal example of using the framework.
#       It should be easy to understand and should not contain any unnecessary code.
#       lightning is not used in the example, so it should be removed.
#       Also, consider removing the wandb logging later.
import logging

import hydra
import lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from nigbms.configs import constructors, meta_solvers, solvers, surrogates, wrapper  # noqa
from nigbms.tasks import MinimizeTestFunctionTask
from omegaconf import OmegaConf
from torchinfo import summary

log = logging.getLogger(__name__)


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
@hydra.main(version_base="1.3", config_path=".", config_name="minimize_testfunctions")
def main(cfg):
    log.info(OmegaConf.to_yaml(cfg))
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
    wrapped_solver = instantiate(cfg.wrapper, solver=solver, surrogate=surrogate, constructor=constructor)
    summary(surrogate)

    for i in range(1, cfg.problem.num_iter + 1):
        # clear gradients
        opt.zero_grad()

        # forward pass
        theta = meta_solver(tau)
        y = wrapped_solver(tau, theta)

        # backprop for theta
        loss(y).backward()

        # logging
        ref = theta.clone()  # copy to get the true gradient
        f_true = torch.autograd.grad(test_func(ref).sum(), ref)[0]
        sim = torch.cosine_similarity(f_true, theta.grad, dim=1, eps=1e-20).detach()
        wandb.log({"ymean": y.mean(), "ymax": y.max(), "ymin": y.min(), "cos_sim": sim.mean()})
        if i % 100 == 0:
            surroate_loss = wrapped_solver.loss_dict["loss"]
            log.info(f"{i=}, {y.mean()=:.3g}, {y.max()=:.3g}, {y.min()=:.3g}, {sim.mean()=:.3g}, {surroate_loss=:.3g}")

        # clip gradients
        if cfg.clip:
            torch.nn.utils.clip_grad_norm_(meta_solver.parameters(), cfg.clip)

        # update
        opt.step()

    return y.mean()


if __name__ == "__main__":
    main()