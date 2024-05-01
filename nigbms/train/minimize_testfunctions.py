import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensordict import TensorDict

import nigbms  # noqa
import wandb
from nigbms.modules.solvers import TestFunctionSolver  # noqa
from nigbms.modules.wrapper import WrappedSolver
from nigbms.utils.resolver import calc_in_channels, calc_indim

OmegaConf.register_new_resolver("calc_indim", calc_indim)
OmegaConf.register_new_resolver("calc_in_channels", calc_in_channels)


### PLOTTING ###
def plot_results(results, test_function, cfg):
    # draw contour lines
    _, ax = plt.subplots(1, 1, figsize=(8, 6))

    plot_params = {
        "f_true": {"color": "r", "alpha": 1.0},
        "f_fwd": {"color": "k", "alpha": 1.0},
        "f_hat_true": {"color": "g", "alpha": 1.0},
        "cv_fwd": {"color": "b", "alpha": 1.0},
    }
    labels = {
        "f_true": r"$\nabla f$",
        "f_fwd": r"$\mathbf{g}_\mathrm{v}$",
        "f_hat_true": r"$\nabla \hat f$",
        "cv_fwd": r"$\mathbf{h}_\mathrm{v}$",
    }
    for grad_type, steps_list in results.items():
        for i, steps in enumerate(steps_list):
            z = test_function(steps)
            ax.plot(
                np.arange(len(z)),
                z.mean(dim=1).ravel(),
                label=labels[grad_type] if i == 0 else None,
                **plot_params[grad_type],
            )
            if grad_type == "f_true":
                ax.set_ylim(z.mean(dim=1).min() * 0.1, z.mean(dim=1).max() * 10)

    ax.set_yscale("log")
    ax.legend()
    plt.savefig(
        f"{cfg.common.run_cfgs.function_name}{cfg.common.run_cfgs.dim}D.png", bbox_inches="tight", pad_inches=0.05
    )


### MAIN ###
@hydra.main(version_base="1.3", config_path="../configs/train", config_name="minimize_testfunctions")
def main(cfg):
    # set up
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.wandb.project, config=wandb.config, mode=cfg.wandb.mode)
    torch.set_default_tensor_type(eval(cfg.problem.tensor_type))
    pl.seed_everything(seed=cfg.seed, workers=True)

    tau = {}
    x = torch.Tensor(cfg.problem.num_samples, cfg.problem.dim).uniform_(*cfg.problem.initial_range)
    x.requires_grad = True
    theta = TensorDict({"x": x})
    solver = instantiate(cfg.solver)
    surrogate = instantiate(cfg.surrogate)
    wrapped_solver = WrappedSolver(solver, surrogate, cfg.wrapper)
    s_loss = instantiate(cfg.loss.s_loss)
    m_opt = instantiate(cfg.optimizer.m_opt, params=[theta["x"]])
    s_opt = instantiate(cfg.optimizer.s_opt, params=surrogate.parameters())

    for i in range(1, cfg.problem.num_iter + 1):
        # clear gradients
        m_opt.zero_grad()
        s_opt.zero_grad()

        # forward pass
        y, y_hat, dvf, dvf_hat = wrapped_solver(tau, theta)

        # backprop for theta
        y.sum().backward(retain_graph=True, inputs=[theta["x"]])

        # backprop for surrogate model
        if cfg.wrapper.grad_type in ["f_hat_true", "cv_fwd"]:
            s_loss(y, y_hat, dvf, dvf_hat)["s_loss"].mean().backward(inputs=list(surrogate.parameters()))

        # logging
        ref = theta["x"].clone()  # copy to get the true gradient
        f_true = torch.autograd.grad(solver.f(ref).sum(), ref)[0]
        sim = torch.cosine_similarity(f_true, theta["x"].grad, dim=1, eps=1e-20).detach()
        wandb.log({"ymean": y.mean(), "ymax": y.max(), "ymin": y.min(), "cos_sim": sim.mean()})
        if i % 100 == 0:
            print(f"{i}, ymean: {y.mean():.3g}, ymax: {y.max():.3g}, ymin: {y.min():.3g}, cos_sim: {sim.mean():.3g}")

        # updatels
        m_opt.step()
        s_opt.step()


if __name__ == "__main__":
    main()
