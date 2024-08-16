import pickle

import hydra
from hydra.utils import instantiate
from petsc4py import PETSc
from tqdm import tqdm

from nigbms.modules.tasks import PETScLinearSystemTask


def save_petsc_task(task: PETScLinearSystemTask, idx: int) -> None:
    viewer_A = PETSc.Viewer().createBinary(f"{idx}_A.dat", "w")
    task.A.view(viewer_A)
    viewer_A.destroy()

    viewer_b = PETSc.Viewer().createBinary(f"{idx}_b.dat", "w")
    task.b.view(viewer_b)
    viewer_b.destroy()

    if task.x is not None:
        viewer_x = PETSc.Viewer().createBinary(f"{idx}_x.dat", "w")
        task.x.view(viewer_x)
        viewer_x.destroy()

    pickle.dump(task.params, open(f"{idx}_params.pkl", "wb"))


@hydra.main(version_base="1.3", config_path="../configs/data", config_name="generate_poisson2d")
def main(cfg) -> None:
    dataset = instantiate(cfg.dataset)
    for i in tqdm(range(cfg.N_data)):
        save_petsc_task(dataset[i], i)


if __name__ == "__main__":
    main()
