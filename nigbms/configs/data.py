from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig

cs = ConfigStore.instance()


@dataclass
class OnlineDatasetConfig:
    """OnlineDatasetConfig class."""

    _target_: str = "nigbms.data.OnlineDataset"
    task_params_type: DictConfig = DictConfig(
        {"_target_": "hydra.utils.get_class", "path": "nigbms.data.generate_poisson2d.Poisson2DParams"}
    )
    task_constructor: DictConfig = DictConfig(
        {"_target_": "hydra.utils.get_method", "path": "nigbms.data.generate_poisson2d.construct_petsc_poisson2d_task"}
    )
    distributions: DictConfig = DictConfig(
        {
            "coef": {"_target_": "nigbms.utils.distributions.NumpyNormal", "shape": [2], "mean": 0, "std": 1},
            "N": {"_target_": "nigbms.utils.distributions.NumpyConstant", "shape": None, "value": 10},
            "degree": {"_target_": "nigbms.utils.distributions.NumpyConstant", "shape": None, "value": 1},
            "rtol": {"_target_": "nigbms.utils.distributions.NumpyConstant", "shape": None, "value": 1.0e-6},
            "maxiter": {"_target_": "nigbms.utils.distributions.NumpyConstant", "shape": None, "value": 1000},
        }
    )


cs.store(name="online_dataset_default", group="data", node=OnlineDatasetConfig)


@dataclass
class OfflineDatasetConfig:
    _target_: str = "nigbms.data.OfflineDataset"
    data_dir: str = "/workspaces/nigbms/data/raw/temp/petsc"
    idcs: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    rtol_dist: DictConfig = DictConfig(
        {"_target_": "nigbms.utils.distributions.NumpyConstant", "shape": None, "value": 1.0e-6}
    )
    maxiter_dist: DictConfig = DictConfig(
        {"_target_": "nigbms.utils.distributions.NumpyConstant", "shape": None, "value": 1000}
    )
    task_type: DictConfig = DictConfig(
        {"_target_": "hydra.utils.get_class", "path": "nigbms.tasks.PETScLinearSystemTask"}
    )
    normalize: bool = False


cs.store(name="offline_dataset_default", group="data", node=OfflineDatasetConfig)


@dataclass
class OfflineDataModuleConfig:
    _target_: str = "nigbms.data.OfflineDataModule"
    data_dir: str = MISSING
    dataset_sizes: DictConfig = MISSING
    rtol_dists: DictConfig = MISSING
    maxiter_dists: DictConfig = MISSING
    in_task_type: DictConfig = MISSING
    out_task_type: DictConfig = MISSING
    batch_size: int = MISSING
    num_workers: int = MISSING
    normalize: bool = MISSING


@dataclass
class Poisson1DOfflineDataModuleConfig(OfflineDataModuleConfig):
    data_dir: str = "/workspaces/nigbms/data/raw/poisson1d/sample"
    dataset_sizes: DictConfig = DictConfig({"train": 100, "val": 100, "test": 100})
    rtol_dists: DictConfig = DictConfig(
        {
            "train": {"_target_": "nigbms.utils.distributions.NumpyConstant", "shape": None, "value": 1.0e-3},
            "val": "${.train}",
            "test": "${.train}",
        }
    )
    maxiter_dists: DictConfig = DictConfig(
        {
            "train": {"_target_": "nigbms.utils.distributions.NumpyConstant", "shape": None, "value": 500},
            "val": "${.train}",
            "test": "${.train}",
        }
    )
    in_task_type: DictConfig = DictConfig(
        {"_target_": "hydra.utils.get_class", "path": "nigbms.tasks.PyTorchLinearSystemTask"}
    )
    out_task_type: DictConfig = DictConfig(
        {"_target_": "hydra.utils.get_class", "path": "nigbms.tasks.PyTorchLinearSystemTask"}
    )
    batch_size: int = 32
    num_workers: int = 0
    normalize: bool = False


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
    in_task_type: DictConfig = DictConfig(
        {"_target_": "hydra.utils.get_class", "path": "nigbms.tasks.PETScLinearSystemTask"}
    )
    out_task_type: DictConfig = DictConfig(
        {"_target_": "hydra.utils.get_class", "path": "nigbms.tasks.PETScLinearSystemTask"}
    )
    batch_size: int = 32
    num_workers: int = 0
    normalize: bool = False


cs.store(name="poisson2d_offline_datamodule", group="data", node=Poisson2DOfflineDataModuleConfig)
