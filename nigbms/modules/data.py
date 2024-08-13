# %%
from typing import Any, Callable, List, Tuple, Union

import pandas as pd
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset

import nigbms  # noqa
import nigbms.data.generate_poisson2d  # noqa
from nigbms.modules.tasks import PyTorchLinearSystemTask
from nigbms.utils.distributions import Distribution


class OfflineDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        meta_df: pd.DataFrame,
        rtol_dist: Distribution,
        maxiter_dist: Distribution,
        data_format: str,
        is_A_fixed: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.meta_df = meta_df
        self.rtol_dist = rtol_dist
        self.maxiter_dist = maxiter_dist
        self.data_format = data_format
        self.fixed_A = self.load(data_dir + "/A") if is_A_fixed else None

    def load(self, path):
        if self.data_format == "pt":
            return torch.load(path + ".pt")
        else:
            raise ValueError(f"Unknown data format: {self.data_format}")

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx):
        if self.fixed_A is not None:
            A = self.fixed_A
        else:
            A = self.load(self.data_dir + f"/{idx}_A")

        b = self.load(self.data_dir + f"/{idx}_b").reshape(-1, 1)
        x = self.load(self.data_dir + f"/{idx}_x").reshape(-1, 1)
        rtol = self.rtol_dist.sample(idx)
        maxiter = self.maxiter_dist.sample(idx)
        params = self.meta_df.iloc[idx].to_dict()
        params["rtol"] = rtol
        params["maxiter"] = maxiter

        tau = PyTorchLinearSystemTask(params, A, b, x, rtol, maxiter)

        return tau


# TODO: Is IterableDataset more appropriate?
# see https://pytorch.org/docs/stable/data.html#iterable-style-datasets
class OnlineDataset(Dataset):
    def __init__(self, task_params_class: str, task_constructor: Callable, distributions: DictConfig) -> None:
        self.task_params_class = eval(task_params_class)
        self.task_constructor = eval(task_constructor)
        self.distributions = distributions

    def __getitem__(self, idx):
        params = {}
        for k, dist in self.distributions.items():
            params[k] = dist.sample(idx)
        task_params = self.task_params_class(**params)
        tau = self.task_constructor(task_params)
        return tau


class OnlineIterableDataset(IterableDataset):
    def __init__(self, task_params_class: str, task_constructor: Callable, distributions: DictConfig) -> None:
        self.task_params_class = eval(task_params_class)
        self.task_constructor = eval(task_constructor)
        self.distributions = distributions

    def __iter__(self):
        params = {}
        for k, dist in self.distributions.items():
            params[k] = dist.sample()
        task_params = self.task_params_class(**params)
        tau = self.task_constructor(task_params)
        return tau


def offline_collate_fn(batch: List[Any]) -> Any:
    batch = [torch.stack(x) for x in zip(*batch, strict=False)]
    tau = PyTorchLinearSystemTask(*batch)
    return tau


class OfflineDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        ds_sizes: List[int],
        fixed_A: bool,
        train_rtol: Union[float, Tuple[float, float]],
        test_rtol: Union[float, Tuple[float, float]],
        train_maxiter: Union[int, Tuple[int, int]],
        test_maxiter: Union[int, Tuple[int, int]],
        batch_size: int,
        num_workers: int,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.ds_sizes = ds_sizes
        self.fixed_A = fixed_A
        self.train_rtol = train_rtol
        self.test_rtol = test_rtol
        self.train_maxiter = train_maxiter
        self.test_maxiter = test_maxiter
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self) -> None:
        meta_df = pd.read_csv(self.data_dir + "/meta_df.csv")
        self.meta_dfs = {
            "train": meta_df.iloc[0 : self.ds_sizes[0]],
            "val": meta_df.iloc[self.ds_sizes[0] : self.ds_sizes[0] + self.ds_sizes[1]],
            "test": meta_df.iloc[
                self.ds_sizes[0] + self.ds_sizes[1] : self.ds_sizes[0] + self.ds_sizes[1] + self.ds_sizes[2]
            ],
        }

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_ds = OfflineDataset(
                self.data_dir, self.meta_dfs["train"], self.fixed_A, self.train_rtol, self.train_maxiter
            )
            self.val_ds = OfflineDataset(
                self.data_dir, self.meta_dfs["val"], self.fixed_A, self.test_rtol, self.test_maxiter
            )

        if stage == "test":
            self.test_ds = OfflineDataset(
                self.data_dir, self.meta_dfs["test"], self.fixed_A, self.test_rtol, self.test_maxiter
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=offline_collate_fn,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=offline_collate_fn,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=offline_collate_fn,
            generator=torch.Generator(device="cuda"),
        )


# %%
