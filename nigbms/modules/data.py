# %%
from dataclasses import astuple
from typing import Any, List, Tuple, Union

import pandas as pd
import torch
from lightning import LightningDataModule
from tensordict import TensorDict
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from nigbms.modules.data import PyTorchLinearSystemTask
from nigbms.utils.distributions import Constant, LogUniform


class OfflineDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        meta_df: pd.DataFrame,
        fixed_A: bool = True,
        rtol: Union[float, Tuple[float, float]] = 1.0e-6,
        maxiter: Union[int, Tuple[int, int]] = 1000,
    ) -> None:
        self.data_dir = data_dir
        self.meta_df = meta_df

        if fixed_A:
            self.fixed_A = torch.load(data_dir + "/A.pt")
            self.fixed_A = self.fixed_A.to_dense()  # TODO: remove this line

        if isinstance(rtol, tuple):
            self.rtol_dist = LogUniform(rtol[0], rtol[1])
        elif isinstance(rtol, float):
            self.rtol_dist = Constant(rtol)
        else:
            raise ValueError("rtol must be a float or a tuple of floats")

        if isinstance(maxiter, tuple):
            self.maxiter_dist = Constant(maxiter[0], maxiter[1])
        elif isinstance(maxiter, int):
            self.maxiter_dist = Constant(maxiter)
        else:
            raise ValueError("maxiter must be an int or a tuple of ints")

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx):
        if isinstance(self.fixed_A, Tensor):
            A = self.fixed_A
        else:
            A = torch.load(self.data_dir + f"/{idx}_A.pt")

        b = torch.load(self.data_dir + f"/{idx}_b.pt").reshape(-1, 1)
        x = torch.load(self.data_dir + f"/{idx}_x.pt").reshape(-1, 1)
        rtol = self.rtol_dist.sample()
        maxiter = self.maxiter_dist.sample().type(torch.int)
        features = TensorDict(
            {
                "rtol": rtol.clone(),
                "maxiter": maxiter.clone(),
                "b": b.clone(),
                "x": x.clone(),
            }
        )

        tau = PyTorchLinearSystemTask(A, b, x, rtol, maxiter, features)

        return astuple(tau)


class OnlineDataset(Dataset):
    def __init__(self, task_generator) -> None:
        self.task_generator = task_generator

    def __getitem__(self, idx):
        tau = self.task_generator(idx)
        return astuple(tau)


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
