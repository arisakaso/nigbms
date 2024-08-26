import os
import subprocess

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb


@hydra.main(version_base="1.3", config_path="../configs/sweep", config_name="poisson1d_small")
def main(cfg: DictConfig):
    wandb.require("core")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["command"] = ["${env}", "python", "${program}", "${args_no_hyphens}"]  # for hydra compatibility
    sweep_id = wandb.sweep(cfg)
    path = wandb.Api().sweep(sweep_id).path
    agents = []
    RUN_PER_GPU = 1

    for i in range(torch.cuda.device_count()):
        for _ in range(RUN_PER_GPU):
            agents.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={i} wandb agent {os.path.join(*path)}", shell=True))

    try:
        for agent in agents:
            agent.wait()
    except KeyboardInterrupt:
        try:
            for agent in agents:
                agent.wait()
        except KeyboardInterrupt:
            subprocess.run(f"wandb sweep --cancel {os.path.join(*path)}")

    print("Sweep finished")


if __name__ == "__main__":
    main()
