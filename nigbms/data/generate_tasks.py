import pickle

import hydra
from joblib import Parallel, delayed
from tqdm import tqdm

from nigbms.modules.data import OnlineDataset


def save_task(task, idx, save_dir) -> None:
    pickle.dump(task, open(f"{save_dir}/{idx}.pkl", "wb"))


@hydra.main(version_base="1.3", config_path="../configs/data", config_name="generate_poisson1d")
def main(cfg) -> None:
    dataset = OnlineDataset(cfg.dataset)
    Parallel(verbose=10, n_jobs=-1)([delayed(save_task)(dataset[i], i, cfg.save_dir) for i in tqdm(cfg.N_data)])
