from pathlib import Path

import hydra
from hydra.utils import instantiate
from tqdm import tqdm

from nigbms.modules.tasks import save_petsc_task


@hydra.main(version_base="1.3", config_path="../configs/data", config_name="generate_poisson2d")
def main(cfg) -> None:
    dataset = instantiate(cfg.dataset)
    for i in tqdm(range(cfg.N_data)):
        save_petsc_task(dataset[i], Path(str(i)))


if __name__ == "__main__":
    main()
