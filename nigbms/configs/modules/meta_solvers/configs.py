from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

cs = ConfigStore.instance()


@dataclass
class ConstantMetaSolverConfig:
    _target_: str = "nigbms.modules.meta_solvers.ConstantMetaSolver"
    params_learn: DictConfig = DictConfig({})
    features: DictConfig = DictConfig({})
    model: DictConfig = DictConfig(
        {"_target_": "nigbms.modules.models.Constant", "shape": [100, 200], "range": [-1, 1]}
    )


cs.store(name="constant_meta_solver_default", node=ConstantMetaSolverConfig)
