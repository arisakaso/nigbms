# %%
from dataclasses import astuple, dataclass
from typing import Any, Dict, List, Tuple, Union

import pandas as pd
import torch
from lightning import LightningDataModule
from tensordict import TensorDict
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from nigbms.utils.distributions import Constant, LogUniform


@dataclass
class Task:
    A: torch.Tensor = None
    b: torch.Tensor = None
    x: torch.Tensor = None
    rtol: torch.Tensor = None
    maxiter: torch.Tensor = None
    features: TensorDict = None


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
        if isinstance(self.fixed_A, torch.Tensor):
            A = self.fixed_A
        else:
            A = torch.load(self.data_dir + f"/{idx}_A.pt")

        b = torch.load(self.data_dir + f"/{idx}_b.pt")
        x = torch.load(self.data_dir + f"/{idx}_x.pt")
        rtol = self.rtol_dist.sample()
        maxiter = self.maxiter_dist.sample()
        features = TensorDict({"rtol": rtol.clone(), "maxiter": maxiter.clone()})

        tau = Task(A, b, x, rtol, maxiter, features)

        return astuple(tau)


def offline_collate_fn(batch: List[Any]) -> Any:
    batch = [torch.stack(x) for x in zip(*batch, strict=False)]
    tau = Task(*batch)
    return tau


class OfflineDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        meta_dfs: Dict[str, pd.DataFrame],
        fixed_A: bool,
        train_rtol: Union[float, Tuple[float, float]],
        test_rtol: Union[float, Tuple[float, float]],
        train_maxiter: Union[int, Tuple[int, int]],
        test_maxiter: Union[int, Tuple[int, int]],
        batch_size: int,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.meta_dfs = meta_dfs
        self.fixed_A = fixed_A
        self.train_rtol = train_rtol
        self.test_rtol = test_rtol
        self.train_maxiter = train_maxiter
        self.test_maxiter = test_maxiter
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        pass

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
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, collate_fn=offline_collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, collate_fn=offline_collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, collate_fn=offline_collate_fn)


# %%
if __name__ == "__main__":
    # memo
    data_dir = "/home/arisaka/nigbms/data/raw/poisson1d/2024-05-06_06-07-05"
    meta_dfs = {
        "train": pd.read_csv(data_dir + "/meta_df.csv"),
        "val": pd.read_csv(data_dir + "/meta_df.csv"),
        "test": pd.read_csv(data_dir + "/meta_df.csv"),
    }
    dm = OfflineDataModule(data_dir, meta_dfs, True, 1.0e-6, 1.0e-6, 1000, 1000, 32)
    dm.setup("fit")
    dl = dm.train_dataloader()
    for batch in dl:
        print(batch)
