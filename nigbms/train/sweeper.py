import os
import subprocess
import time

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../configs/sweep", config_name="poisson1d_small")
def main(cfg: DictConfig):
    RUN_PER_GPU = 2
    path = "sohei/poisson1d_small/9h7gji5c"

    if path is None:
        wandb.require("core")
        cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg["command"] = ["${env}", "python", "${program}", "${args_no_hyphens}"]  # for hydra compatibility
        sweep_id = wandb.sweep(cfg)
        path = wandb.Api().sweep(sweep_id).path
        path = os.path.join(*path)

    agents = []
    for i in range(torch.cuda.device_count()):
        for _ in range(RUN_PER_GPU):
            agents.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={i} wandb agent {path}", shell=True))
            time.sleep(1)
    try:
        for agent in agents:
            agent.wait()
    except KeyboardInterrupt:
        try:
            for agent in agents:
                agent.wait()
        except KeyboardInterrupt:
            subprocess.run(f"wandb sweep --cancel {path}")

    print("Sweep finished")


if __name__ == "__main__":
    main()
