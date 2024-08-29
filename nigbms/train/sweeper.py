import os
import subprocess
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import wandb


def create_new_sweep(cfg: DictConfig) -> str:
    wandb.require("core")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["command"] = ["${env}", "python", "${program}", "${args_no_hyphens}"]  # for hydra compatibility
    sweep_id = wandb.sweep(cfg)
    path = wandb.Api().sweep(sweep_id).path
    return os.path.join(*path)


@hydra.main(version_base="1.3", config_path="../configs/sweep", config_name="poisson1d_small")
def main(cfg: DictConfig):
    RUN_PER_GPU = 2
    # path = "sohei/poisson1d_small/9h7gji5c"
    path = None

    if path is None:
        path = create_new_sweep(cfg)

    agents = []
    for i in range(torch.cuda.device_count()):
        for _ in range(RUN_PER_GPU):
            agents.append(
                # start_new_session seems help to avoid wandb connection error (I'm not sure)
                subprocess.Popen(f"CUDA_VISIBLE_DEVICES={i} wandb agent {path}", shell=True, start_new_session=True)
            )
            time.sleep(1)

    # Wait for all agents to finish
    try:
        for agent in agents:
            agent.wait()
    # Two levels of KeyboardInterrupt handling
    except KeyboardInterrupt:
        # stop creating new runs
        subprocess.run(f"wandb sweep --stop {path}", shell=True)
        try:
            for agent in agents:
                agent.wait()
        except KeyboardInterrupt:
            time.sleep(5)
            # kill all runs and finish the sweep
            subprocess.run(f"wandb sweep --cancel {path}", shell=True)

    print("Sweep finished")


if __name__ == "__main__":
    main()
