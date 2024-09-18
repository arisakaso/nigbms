import warnings
from pathlib import Path
from typing import Callable, Dict, List, Type

import numpy as np
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset

from nigbms.tasks import (
    PETScLinearSystemTask,
    PyTorchLinearSystemTask,
    Task,
    TaskConstructor,
    TaskDistribution,
    load_petsc_task,
    load_pytorch_task,
    petsc2torch_collate_fn,
    petsc_task_collate_fn,
    pytorch_task_collate_fn,
    torch2petsc_collate_fn,
)
from nigbms.utils.distributions import Distribution

# suppress warnings. TODO: Deal with multiple workers compatibility
warnings.filterwarnings("ignore", ".*does not have many workers.*")


class OfflineDataset(Dataset):
    """Offline dataset for loading pre-generated tasks from disk."""

    def __init__(
        self,
        data_dir: Path,
        idcs: List[int],
        rtol_dist: Distribution,
        maxiter_dist: Distribution,
        task_type: Type,
        normalize: bool = False,
    ) -> None:
        """

        Args:
            data_dir (Path): directory containing the pre-generated tasks.
                Each task is saved in a subdirectory with the task id as the name.
            idcs (List[int]): list of task ids to load.
            rtol_dist (Distribution): distribution for rtol values.
            maxiter_dist (Distribution): distribution for maxiter values.
            task_type (Type): task type, either PyTorchLinearSystemTask or PETScLinearSystemTask.
        """
        self.data_dir = Path(data_dir)
        self.idcs = idcs
        self.rtol_dist = rtol_dist
        self.maxiter_dist = maxiter_dist
        self.task_type = task_type
        self.normalize = normalize

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
        if self.normalize:
            tau.b = tau.b / tau.b.norm(dim=0, keepdim=True)
        return tau


# TODO: Is IterableDataset more appropriate?
# see https://pytorch.org/docs/stable/data.html#iterable-style-datasets
class OnlineDataset(Dataset):
    """Online dataset for generating tasks on-the-fly."""

    def __init__(self, constructor: TaskConstructor, distribution: TaskDistribution) -> None:
        """
        Args:
            constructor (TaskConstructor):
            distribution (TaskDistribution):
        """

        self.constructor = constructor
        self.distribution = distribution

    def __getitem__(self, idx) -> Task:
        task_params = self.distribution.sample()
        tau = self.constructor(task_params)
        return tau


class OnlineIterableDataset(IterableDataset):
    """Online iterable dataset for generating tasks on-the-fly."""

    def __init__(self, task_params_type: Type, task_constructor: Callable, distributions: DictConfig) -> None:
        """
        Args:
            task_params_type (Type): TaskParams type
            task_constructor (Callable): task constructor function that takes a TaskParams object and returns a Task
            distributions (DictConfig): dictionary of distributions for each parameter in the TaskParams
        """
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


class OfflineDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        dataset_sizes: Dict[str, int],
        rtol_dists: Dict[str, Distribution],
        maxiter_dists: Dict[str, Distribution],
        in_task_type: Type,
        out_task_type: Type,
        batch_size: int,
        num_workers: int,
        normalize: bool,
    ) -> None:
        """

        Args:
            data_dir (Path): root directory of the dataset
            dataset_sizes (Dict[str, int]): number of samples in each dataset.
                keys: "train", "val", "test"
            rtol_dists (Dict[str, Distribution]): rtol distributions for each dataset.
                keys: "train", "val", "test"
            maxiter_dists (Dict[str, Distribution]): maxiter distributions for each dataset.
                keys: "train", "val", "test"
            in_task_type (Type): task type used for loading the dataset
            out_task_type (Type): task type used for the output of the dataloader
            batch_size (int): batch size
            num_workers (int): number of workers for the dataloader
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dataset_sizes = dataset_sizes
        self.rtol_dists = rtol_dists
        self.maxiter_dits = maxiter_dists
        self.in_task_type = in_task_type
        self.out_task_type = out_task_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize

    def prepare_data(self) -> None:
        """Prepare data by splitting the indices for each dataset."""
        dataset_names, dataset_sizes = zip(*self.dataset_sizes.items(), strict=False)
        indices_ranges = np.split(np.arange(sum(dataset_sizes)), np.cumsum(dataset_sizes)[:-1])
        self.indcs = dict(zip(dataset_names, indices_ranges, strict=False))

    def setup(self, stage: str = None):
        """Choose appropriate collate_fn and set up datasets."""

        # set up collate_fn
        task_combination = (self.in_task_type, self.out_task_type)
        if task_combination == (PyTorchLinearSystemTask, PyTorchLinearSystemTask):
            self.collate_fn = pytorch_task_collate_fn
        elif task_combination == (PyTorchLinearSystemTask, PETScLinearSystemTask):
            self.collate_fn = torch2petsc_collate_fn
        elif task_combination == (PETScLinearSystemTask, PETScLinearSystemTask):
            self.collate_fn = petsc_task_collate_fn
        elif task_combination == (PETScLinearSystemTask, PyTorchLinearSystemTask):
            self.collate_fn = petsc2torch_collate_fn
        else:
            raise ValueError(f"Unsupported task combination: {task_combination}")

        # setup datasets
        if stage == "fit" or stage is None:
            self.train_ds = OfflineDataset(
                self.data_dir,
                self.indcs["train"],
                self.rtol_dists["train"],
                self.maxiter_dits["train"],
                self.in_task_type,
                self.normalize,
            )
            self.val_ds = OfflineDataset(
                self.data_dir,
                self.indcs["val"],
                self.rtol_dists["val"],
                self.maxiter_dits["val"],
                self.in_task_type,
                self.normalize,
            )

        if stage == "test":
            self.test_ds = OfflineDataset(
                self.data_dir,
                self.indcs["test"],
                self.rtol_dists["test"],
                self.maxiter_dits["test"],
                self.in_task_type,
                self.normalize,
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
            num_workers=self.num_workers,
            # generator=torch.Generator(device="cuda"), # What was this...?
        )