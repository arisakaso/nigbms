import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensordict import TensorDict

import nigbms  # noqa
from nigbms.modules.solvers import TestFunctionSolver  # noqa
from nigbms.modules.wrapper import WrappedSolver
from nigbms.utils.resolver import calc_indim

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("calc_indim", calc_indim)


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
    log.info(cfg.wrapper.grad_type)
    log.info(os.getcwd())

    # set up
    pl.seed_everything(seed=cfg.seed, workers=True)
    torch.set_default_tensor_type(eval(cfg.problem.tensor_type))
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
    ys = torch.zeros((cfg.problem.num_iter + 1, cfg.problem.num_samples))
    sims = torch.zeros((cfg.problem.num_iter + 1, cfg.problem.num_samples))
    ys[0] = solver.f(theta["x"].detach()).squeeze()

    for i in range(1, cfg.problem.num_iter + 1):
        m_opt.zero_grad()
        s_opt.zero_grad()

        # reference for checking cosine similarity
        ref = theta["x"].clone()
        f_true = torch.autograd.grad(solver.f(ref).sum(), ref)[0]

        # forward pass
        y, y_hat, dvf, dvf_hat = wrapped_solver(tau, theta)

        # backprop for theta
        y.sum().backward(retain_graph=True)

        # backprop for surrogate model
        s_loss(y, y_hat, dvf, dvf_hat)["s_loss"].mean().backward()

        # update
        m_opt.step()
        s_opt.step()

        # store steps
        ys[i] = y.detach().squeeze()
        sims[i] = torch.cosine_similarity(f_true, theta["x"].grad, dim=1, eps=1e-20).detach()

        # print progress
        if i % 100 == 0:
            log.info(
                f"{i}: ymean={y.mean():.3g}, ymax={y.max():.3g}, ymed={y.median():.3g}, ymin={y.min():.3g}, sim={sims[i].mean():.3g}"
            )

    ys = ys.cpu()
    sims = sims.cpu()
    torch.save(ys, f"{cfg.problem.test_function}{cfg.problem.dim}D-{cfg.wrapper.grad_type}-y.pt")
    torch.save(sims, f"{cfg.problem.test_function}{cfg.problem.dim}D-{cfg.wrapper.grad_type}-sim.pt")


if __name__ == "__main__":
    main()
