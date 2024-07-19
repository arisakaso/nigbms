# %%
import pickle
from typing import List

import pandas as pd
import pytorch_lightning as pl
import torch
from scipy.sparse import diags
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from src.utils.utils import Constant, LogUniform


# %%
def get_A(N=128, with_boundary=False, dtype=torch.float32):
    """create A matrix corresponding to poisson equation

    Args:
        bc_type (str, optional): [description]. Defaults to "d".
        N (int, optional): [description]. Defaults to 128.

    Returns:
        [type]: [description]
    """

    if with_boundary:
        A = torch.Tensor(diags([1, -2, 1], [-1, 0, 1], shape=(N + 2, N + 2)).toarray())
        A[0, 0] = 1
        A[0, 1] = 0
        A[-1, -1] = 1
        A[-1, -2] = 0
    else:
        A = torch.Tensor(diags([1, -2, 1], [-1, 0, 1], shape=(N, N)).toarray())

    return A.type(dtype)


#### DATASET ####
# %%
class PoissonDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        meta_df: pd.DataFrame,
        normalize_type: str = None,
        stats_df: pd.DataFrame = None,
        rtol: List[float] = None,
        maxiter: int = None,
        dtype=torch.float32,
        dim: int = 32,
        D: int = 1,
        sparse: bool = False,
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
        self.dim = dim
        if D == 1:
            self.A = get_A(N=dim, dtype=dtype)
        elif D == 2:
            self.A = torch.load(f"{root_dir}/A.pt").type(dtype)
        if sparse:
            self.A = self.A.to_sparse_coo()

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx):
        _idx = self.meta_df.index[idx]

        A = -self.A
        b, x = torch.load(f"{self.root_dir}/{_idx}_fu.pt")
        b = -b.type(self.dtype)
        x = x.type(self.dtype)
        rtol = self.rtol.sample()
        tau = {
            "A": A,
            "b": b,
            "x_sol": x,
            "rtol": rtol,
            "maxiter": self.maxiter,
        }

        return tau


class Poisson1DPrecomputedDataset(Dataset):
    def __init__(self, root_dir, id, length, dtype=torch.float32):
        self.root_dir = root_dir
        self.id = id
        self.dtype = dtype
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        epoch = idx // 100
        batch_idx = idx % 100
        batch = pickle.load(
            open(f"{self.root_dir}/{self.id}/{epoch}_{batch_idx}.pkl", "rb")
        )
        batch = [t.type(self.dtype) for t in batch]
        return batch


class PoissonDataModule(pl.LightningDataModule):
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
        dim: int = 32,
        D: int = 1,
        sparse: bool = False,
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
        self.dim = dim
        self.D = D
        self.sparse = sparse

    def train_dataloader(self):
        return DataLoader(
            PoissonDataset(
                self.root_dir,
                self.meta_dfs["train"],
                self.normalize_type,
                self.stats_df,
                [self.rtol[0], self.rtol[0]],
                self.maxiter,
                self.dtype,
                self.dim,
                self.D,
                self.sparse,
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return [
            DataLoader(
                PoissonDataset(
                    self.root_dir,
                    self.meta_dfs["val"],
                    self.normalize_type,
                    self.stats_df,
                    [self.rtol[0], self.rtol[0]],
                    self.maxiter_test,
                    self.dtype,
                    self.dim,
                    self.D,
                    self.sparse,
                ),
                batch_size=self.batch_size,
                shuffle=False,
            ),
            DataLoader(
                PoissonDataset(
                    self.root_dir,
                    self.meta_dfs["train"],
                    self.normalize_type,
                    self.stats_df,
                    [self.rtol[0], self.rtol[0]],
                    self.maxiter_test,
                    self.dtype,
                    self.dim,
                    self.D,
                    self.sparse,
                ),
                batch_size=self.batch_size,
                shuffle=False,
            ),
        ]

    def test_dataloader(self):
        return DataLoader(
            PoissonDataset(
                self.root_dir,
                self.meta_dfs["test"],
                self.normalize_type,
                self.stats_df,
                [self.rtol[0], self.rtol[0]],
                self.maxiter_test,
                self.dtype,
                self.dim,
                self.D,
                self.sparse,
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )


# %%
