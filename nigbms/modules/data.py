# %%
from pathlib import Path
from typing import Callable, Dict, List, Type

import numpy as np
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset

from nigbms.modules.tasks import (
    PETScLinearSystemTask,
    PyTorchLinearSystemTask,
    Task,
    load_petsc_task,
    load_pytorch_task,
)
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

    def __getitem__(self, idx) -> Task:
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

    def __iter__(self) -> Task:
        params = {}
        for k, dist in self.distributions.items():
            params[k] = dist.sample()
        task_params = self.task_params_type(**params)
        tau = self.task_constructor(task_params)
        return tau


def pytorch_task_collate_fn(batch: List[PyTorchLinearSystemTask]) -> PyTorchLinearSystemTask:
    tau = torch.stack(batch)
    tau.is_batched = True
    return tau


def petsc_task_collate_fn(batch: List[PETScLinearSystemTask]) -> List[PETScLinearSystemTask]:
    return batch


class OfflineDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        dataset_sizes: Dict[str, int],
        rtol_dists: Dict[str, Distribution],
        maxiter_dists: Dict[str, Distribution],
        task_type: Type,
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_sizes = dataset_sizes
        self.rtol_dists = rtol_dists
        self.maxiter_dits = maxiter_dists
        self.task_type = task_type
        if task_type == PyTorchLinearSystemTask:
            self.collate_fn = pytorch_task_collate_fn
        elif task_type == PETScLinearSystemTask:
            self.collate_fn = petsc_task_collate_fn
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        dataset_names, dataset_sizes = zip(*self.dataset_sizes.items(), strict=False)
        indices_ranges = np.split(np.arange(sum(dataset_sizes)), np.cumsum(dataset_sizes)[:-1])
        self.indcs = dict(zip(dataset_names, indices_ranges, strict=False))

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_ds = OfflineDataset(
                self.data_dir,
                self.indcs["train"],
                self.rtol_dists["train"],
                self.maxiter_dits["train"],
                self.task_type,
            )
            self.val_ds = OfflineDataset(
                self.data_dir,
                self.indcs["val"],
                self.rtol_dists["val"],
                self.maxiter_dits["val"],
                self.task_type,
            )

        if stage == "test":
            self.test_ds = OfflineDataset(
                self.data_dir,
                self.indcs["test"],
                self.rtol_dists["test"],
                self.maxiter_dits["test"],
                self.task_type,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            generator=torch.Generator(device="cuda"),
        )


# %%
