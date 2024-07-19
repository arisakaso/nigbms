# %%
from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from src.utils.utils import Constant, LogUniform


#### DATASET ####
# %%
class BeamDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        meta_df: pd.DataFrame,
        normalize_type: str = None,
        stats_df: pd.DataFrame = None,
        rtol: List[float] = None,
        maxiter: int = None,
        dtype=torch.float32,
    ) -> None:
        self.root_dir = root_dir
        self.meta_df = meta_df
        self.normalize_type = normalize_type
        self.stats_df = stats_df
        if rtol[0] < rtol[1]:
            self.rtol = LogUniform(rtol[0], rtol[1])
        elif rtol[0] == rtol[1]:
            self.rtol = Constant(rtol[0])
        else:
            raise ValueError("rtol[0] must be smaller than rtol[1]")
        self.maxiter = torch.tensor(maxiter)
        self.dtype = dtype
        self.A = torch.load(self.root_dir + "/0_A.pt")

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx):
        _idx = self.meta_df.index[idx]
        A = self.A  # torch.load(self.root_dir + f"/{idx}_A.pt")
        b, x = torch.load(self.root_dir + f"/{_idx}_fu.pt")
        rtol = self.rtol.sample()
        features = torch.tensor(self.meta_df.iloc[idx], dtype=self.dtype)

        tau = {
            "A": A,
            "b": b,
            "x_sol": x,
            "rtol": rtol,
            "maxiter": self.maxiter,
            "features": features,
        }

        return tau


class BeamDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        n_samples: List[int],
        batch_size: int,
        normalize_type: str = None,
        rtol: List[float] = None,
        maxiter: int = 1000,
        maxiter_test: int = 1000,
        dtype: str = "torch.float32",
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        meta_df_all = pd.read_csv(root_dir + "/meta_df.csv")
        self.meta_dfs = {
            "train": meta_df_all.iloc[0 : n_samples[0]],
            "val": meta_df_all.iloc[n_samples[0] : n_samples[0] + n_samples[1]],
            "test": meta_df_all.iloc[
                n_samples[0] + n_samples[1] : n_samples[0] + n_samples[1] + n_samples[2]
            ],
        }
        self.normalize_type = normalize_type
        if self.normalize_type is not None:
            self.stats_df = pd.read_csv(root_dir + "/stats_df.csv", index_col=0)
        else:
            self.stats_df = None
        self.rtol = rtol
        self.maxiter = maxiter
        self.maxiter_test = maxiter_test
        self.dtype = eval(dtype)

    def train_dataloader(self):
        return DataLoader(
            BeamDataset(
                self.root_dir,
                self.meta_dfs["train"],
                self.normalize_type,
                self.stats_df,
                [self.rtol[0], self.rtol[0]],
                self.maxiter,
                self.dtype,
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                BeamDataset(
                    self.root_dir,
                    self.meta_dfs["train"],
                    self.normalize_type,
                    self.stats_df,
                    [self.rtol[0], self.rtol[0]],
                    self.maxiter_test,
                    self.dtype,
                ),
                batch_size=self.batch_size,
                shuffle=True,
            ),
            DataLoader(
                BeamDataset(
                    self.root_dir,
                    self.meta_dfs["val"],
                    self.normalize_type,
                    self.stats_df,
                    [self.rtol[0], self.rtol[0]],
                    self.maxiter_test,
                    self.dtype,
                ),
                batch_size=self.batch_size,
                shuffle=False,
            ),
        ]

    def test_dataloader(self):
        return [
            DataLoader(
                BeamDataset(
                    self.root_dir,
                    self.meta_dfs["test"],
                    self.normalize_type,
                    self.stats_df,
                    [rtol, rtol],
                    self.maxiter_test,
                    self.dtype,
                ),
                batch_size=self.batch_size,
                shuffle=False,
            )
            for rtol in [1e-2, 1e-3, 1e-4, 1e-5]
        ]
