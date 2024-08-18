from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig


@dataclass
class PETScKSPConfig:
    _target_: str = "nigbms.modules.solvers.PETScKSP"
    params_fix: DictConfig = DictConfig({"history_length": 100})
    params_learn: DictConfig = DictConfig({})
    debug: bool = True


cs = ConfigStore.instance()
cs.store(name="petsc_ksp_default", node=PETScKSPConfig)
