# %%
from dataclasses import astuple
from pathlib import Path
from typing import Callable, Dict, List, Type

import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset

from nigbms.modules.tasks import PETScLinearSystemTask, PyTorchLinearSystemTask, load_petsc_task, load_pytorch_task
from nigbms.utils.distributions import Distribution


class OfflineDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        idcs: List[int],
        rtol_dist: Distribution,
        maxiter_dist: Distribution,
        task_type: Type,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.idcs = idcs
        self.rtol_dist = rtol_dist
        self.maxiter_dist = maxiter_dist
        self.task_type = task_type

    def load(self, path: Path) -> PyTorchLinearSystemTask | PETScLinearSystemTask:
        if self.task_type == PyTorchLinearSystemTask:
            return load_pytorch_task(path)
        elif self.task_type == PETScLinearSystemTask:
            return load_petsc_task(path)

    def __len__(self) -> int:
        return len(self.idcs)

    def __getitem__(self, idx) -> PyTorchLinearSystemTask | PETScLinearSystemTask:
        tau = self.load(self.data_dir / str(self.idcs[idx]))
        tau.rtol = self.rtol_dist.sample()
        tau.maxiter = self.maxiter_dist.sample()
        return tau


# TODO: Is IterableDataset more appropriate?
# see https://pytorch.org/docs/stable/data.html#iterable-style-datasets
class OnlineDataset(Dataset):
    def __init__(self, task_params_type: Type, task_constructor: Callable, distributions: DictConfig) -> None:
        self.task_params_type = task_params_type
        self.task_constructor = task_constructor
        self.distributions = distributions

    def __getitem__(self, idx):
        params = {}
        for k, dist in self.distributions.items():
            params[k] = dist.sample(idx)
        task_params = self.task_params_type(**params)
        tau = self.task_constructor(task_params)
        return tau


class OnlineIterableDataset(IterableDataset):
    def __init__(self, task_params_type: Type, task_constructor: Callable, distributions: DictConfig) -> None:
        self.task_params_type = task_params_type
        self.task_constructor = task_constructor
        self.distributions = distributions

    def __iter__(self):
        params = {}
        for k, dist in self.distributions.items():
            params[k] = dist.sample()
        task_params = self.task_params_type(**params)
        tau = self.task_constructor(task_params)
        return tau


def pytorch_task_collate_fn(batch: List[PyTorchLinearSystemTask]) -> PyTorchLinearSystemTask:
    batch = [astuple(tau) for tau in batch]
    batch = [torch.stack(x) for x in zip(*batch, strict=False)]
    tau = PyTorchLinearSystemTask(*batch)
    return tau


class OfflineDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        dataset_sizes: Dict[str, int],
        rtol_dists: Dict[str, Distribution],
        maxiter_dists: Dict[str, Distribution],
        data_format: str,
        is_A_fixed: bool,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.dataset_sizes = dataset_sizes
        self.rtol_dists = rtol_dists
        self.maxiter_dits = maxiter_dists
        self.data_format = data_format
        self.is_A_fixed = is_A_fixed
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        meta_df = pd.read_csv(self.data_dir + "/meta_df.csv")
        keys, sizes = zip(*self.dataset_sizes.items(), strict=False)
        self.meta_dfs = dict(zip(keys, np.split(meta_df, sizes), strict=False))

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_ds = OfflineDataset(
                self.data_dir,
                self.meta_dfs["train"],
                self.rtol_dists["train"],
                self.maxiter_dits["train"],
                self.data_format,
                self.is_A_fixed,
            )
            self.val_ds = OfflineDataset(
                self.data_dir,
                self.meta_dfs["val"],
                self.rtol_dists["val"],
                self.maxiter_dits["val"],
                self.data_format,
                self.is_A_fixed,
            )

        if stage == "test":
            self.test_ds = OfflineDataset(
                self.data_dir,
                self.meta_dfs["test"],
                self.rtol_dists["test"],
                self.maxiter_dits["test"],
                self.data_format,
                self.is_A_fixed,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=pytorch_task_collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=pytorch_task_collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=pytorch_task_collate_fn,
            generator=torch.Generator(device="cuda"),
        )


# %%
