from dataclasses import dataclass
from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig

cs = ConfigStore.instance()


@dataclass
class OnlineDatasetConfig:
    _target_: str = "nigbms.modules.data.OnlineDataset"
    task_params_type: DictConfig = MISSING
    task_constructor: DictConfig = MISSING
    distributions: DictConfig = MISSING


@dataclass
class OfflineDatasetConfig:
    _target_: str = "nigbms.modules.data.OfflineDataset"
    data_dir: str = MISSING
    idcs: List[int] = MISSING
    rtol_dist: DictConfig = MISSING
    maxiter_dist: DictConfig = MISSING
    task_type: DictConfig = MISSING


@dataclass
class OfflineDataModuleConfig:
    _target_: str = "nigbms.modules.data.OfflineDataModule"
    data_dir: str = MISSING
    dataset_sizes: DictConfig = MISSING
    rtol_dists: DictConfig = MISSING
    maxiter_dists: DictConfig = MISSING
    task_type: DictConfig = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING


@dataclass
class Poisson1DOfflineDataModuleConfig(OfflineDataModuleConfig):
    data_dir: str = "/workspaces/nigbms/data/raw/poisson1d/sample"
    dataset_sizes: DictConfig = DictConfig({"train": 100, "val": 100, "test": 100})
    rtol_dists: DictConfig = DictConfig(
        {
            "train": {"_target_": "nigbms.utils.distributions.NumpyConstant", "shape": None, "value": 1.0e-6},
            "val": "${.train}",
            "test": "${.train}",
        }
    )
    maxiter_dists: DictConfig = DictConfig(
        {
            "train": {"_target_": "nigbms.utils.distributions.NumpyConstant", "shape": None, "value": 1000},
            "val": "${.train}",
            "test": "${.train}",
        }
    )
    task_type: DictConfig = DictConfig(
        {"_target_": "hydra.utils.get_class", "path": "nigbms.modules.tasks.PyTorchLinearSystemTask"}
    )
    batch_size: int = 32
    num_workers: int = 0


cs.store(name="poisson1d_offline_datamodule", group="data", node=Poisson1DOfflineDataModuleConfig)


@dataclass
class Poisson2DOfflineDataModuleConfig(OfflineDataModuleConfig):
    data_dir: str = "/workspaces/nigbms/data/raw/poisson2d/sample"
    dataset_sizes: DictConfig = DictConfig({"train": 100, "val": 100, "test": 100})
    rtol_dists: DictConfig = DictConfig(
        {
            "train": {"_target_": "nigbms.utils.distributions.NumpyConstant", "shape": None, "value": 1.0e-6},
            "val": "${.train}",
            "test": "${.train}",
        }
    )
    maxiter_dists: DictConfig = DictConfig(
        {
            "train": {"_target_": "nigbms.utils.distributions.NumpyConstant", "shape": None, "value": 1000},
            "val": "${.train}",
            "test": "${.train}",
        }
    )
    task_type: DictConfig = DictConfig(
        {"_target_": "hydra.utils.get_class", "path": "nigbms.modules.tasks.PETScLinearSystemTask"}
    )
    batch_size: int = 32
    num_workers: int = 0


cs.store(name="poisson2d_offline_datamodule", group="data", node=Poisson2DOfflineDataModuleConfig)
