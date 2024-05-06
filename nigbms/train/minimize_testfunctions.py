import hydra
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensordict import TensorDict

import wandb
from nigbms.modules.wrapper import WrappedSolver
from nigbms.utils.resolver import calc_in_channels, calc_in_dim

OmegaConf.register_new_resolver("calc_in_dim", calc_in_dim)
OmegaConf.register_new_resolver("calc_in_channels", calc_in_channels)


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
    s_loss = instantiate(cfg.loss.s_loss)
    m_loss = torch.sum
    s_opt = instantiate(cfg.optimizer.s_opt, params=surrogate.parameters())
    m_opt = instantiate(cfg.optimizer.m_opt, params=[theta["x"]])
    wrapped_solver = WrappedSolver(solver, surrogate, s_opt, s_loss, cfg.optimizer.s_clip, cfg.wrapper)

    for i in range(1, cfg.problem.num_iter + 1):
        # clear gradients
        m_opt.zero_grad()

        # forward pass
        y = wrapped_solver(tau, theta)

        # backprop for theta
        m_loss(y).backward(inputs=[theta["x"]], create_graph=True)

        # logging
        ref = theta["x"].clone()  # copy to get the true gradient
        f_true = torch.autograd.grad(solver.f(ref).sum(), ref)[0]
        sim = torch.cosine_similarity(f_true, theta["x"].grad, dim=1, eps=1e-20).detach()
        wandb.log({"ymean": y.mean(), "ymax": y.max(), "ymin": y.min(), "cos_sim": sim.mean()})
        if i % 100 == 0:
            print(f"{i}, ymean: {y.mean():.3g}, ymax: {y.max():.3g}, ymin: {y.min():.3g}, cos_sim: {sim.mean():.3g}")

        # clip gradients
        if cfg.optimizer.m_clip:
            torch.nn.utils.clip_grad_norm_(theta["x"], cfg.optimizer.m_clip)

        # updates
        m_opt.step()


if __name__ == "__main__":
    main()
