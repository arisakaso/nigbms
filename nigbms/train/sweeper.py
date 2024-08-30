import os
import subprocess
import time

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

# TODO: nested parameters do not work properly with wandb 0.17.7
# It causes duplicate runs in the sweep
# Probably it is wandb's bug, but I'm not sure
# Please use the flatten version of the sweep config
# e.g.
# ```yaml
# parameters:
#   param_group:
#     parameters:
#       param1:
#         values: [1, 2, 3]```
# should be
# ```yaml
# parameters:
#   param_group.param1:
#     values: [1, 2, 3]```
# Although it causes duplicate parameters in the wandb log dashboard, we have no choice for now


def create_new_sweep(cfg: DictConfig) -> str:
    wandb.require("core")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["command"] = ["${env}", "python", "${program}", "${args_no_hyphens}"]  # for hydra compatibility
    sweep_id = wandb.sweep(cfg)
    path = wandb.Api().sweep(sweep_id).path
    return os.path.join(*path)


@hydra.main(version_base="1.3", config_path="../configs/sweep", config_name="poisson1d_small")
def main(cfg: DictConfig):
    path = cfg.settings.sweep_path
    if path is None:
        path = create_new_sweep(cfg.sweep)
    agents = []
    for i in range(torch.cuda.device_count()):
        for _ in range(cfg.settings.agents_per_gpu):
            agents.append(
                # start_new_session seems help to avoid wandb connection error (I'm not sure)
                subprocess.Popen(f"CUDA_VISIBLE_DEVICES={i} wandb agent {path}", shell=True)
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
