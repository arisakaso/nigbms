from pathlib import Path

import hydra
from hydra.utils import instantiate
from nigbms.tasks import save_petsc_task, save_pytorch_task
from tqdm import tqdm


@hydra.main(version_base="1.3", config_path="../configs/data", config_name="generate_poisson1d_large")
def main(cfg) -> None:
    dataset = instantiate(cfg.dataset)
    for i in tqdm(range(cfg.N_data)):
        if "petsc" in cfg.dataset.task_constructor.path:
            save_petsc_task(dataset[i], Path(str(i)))
        elif "pytorch" in cfg.dataset.task_constructor.path:
            save_pytorch_task(dataset[i], Path(str(i)))
        else:
            raise ValueError("Unknown task constructor.")


if __name__ == "__main__":
    main()
