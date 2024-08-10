import hydra
import pytorch_lightning as pl
import torch
import wandb
from hydra.utils import instantiate
from omegaconf import OmegaConf

from nigbms.modules.data import PyTorchTask
from nigbms.modules.wrapper import WrappedSolver
from nigbms.utils.resolver import calc_in_channels, calc_in_dim

OmegaConf.register_new_resolver("calc_in_dim", calc_in_dim)
OmegaConf.register_new_resolver("calc_in_channels", calc_in_channels)
OmegaConf.register_new_resolver("eval", eval)


### MAIN ###
@hydra.main(version_base="1.3", config_path="../configs/train", config_name="minimize_testfunctions")
def main(cfg):
    # set up
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.wandb.project, config=wandb.config, mode=cfg.wandb.mode)
    torch.set_default_dtype(eval(cfg.dtype))
    torch.set_default_device(cfg.device)
    pl.seed_everything(seed=cfg.seed, workers=True)

    tau = PyTorchTask()
    theta = torch.distributions.Uniform(*cfg.problem.initial_range).sample((cfg.problem.num_samples, cfg.problem.dim))
    theta.requires_grad = True
    solver = instantiate(cfg.solver)
    surrogate = instantiate(cfg.surrogate)
    constructor = instantiate(cfg.constructor)
    loss = torch.sum
    opt = instantiate(cfg.opt, params=[theta])
    wrapped_solver = WrappedSolver(solver=solver, surrogate=surrogate, constructor=constructor, **cfg.wrapper)

    for i in range(1, cfg.problem.num_iter + 1):
        # clear gradients
        opt.zero_grad()

        # forward pass
        y = wrapped_solver(tau, theta)

        # backprop for theta
        loss(y).backward(inputs=theta, create_graph=True)

        # logging
        ref = theta.clone()  # copy to get the true gradient
        f_true = torch.autograd.grad(solver.f(ref).sum(), ref)[0]
        sim = torch.cosine_similarity(f_true, theta.grad, dim=1, eps=1e-20).detach()
        wandb.log({"ymean": y.mean(), "ymax": y.max(), "ymin": y.min(), "cos_sim": sim.mean()})
        if i % 100 == 0:
            print(f"{i}, ymean: {y.mean():.3g}, ymax: {y.max():.3g}, ymin: {y.min():.3g}, cos_sim: {sim.mean():.3g}")

        # clip gradients
        if cfg.clip:
            torch.nn.utils.clip_grad_norm_(theta["x"], cfg.clip)

        # update
        opt.step()


if __name__ == "__main__":
    main()
