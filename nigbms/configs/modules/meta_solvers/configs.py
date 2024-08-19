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
        {
            "_target_": "nigbms.modules.models.Constant",
            "shape": [100, 200],
            "range": [-1, 1],
        }
    )


cs.store(name="constant_meta_solver_default", node=ConstantMetaSolverConfig)


@dataclass
class Poisson1DMetaSolverConfig:
    _target_: str = "nigbms.modules.meta_solvers.Poisson1DMetaSolver"
    params_learn: DictConfig = DictConfig({})
    features: DictConfig = DictConfig({"b": 5})
    model: DictConfig = DictConfig(
        {
            "_target_": "nigbms.modules.models.MLP",
            "in_dim": 5,
            "out_dim": 5,
            "num_layers": 3,
            "num_neurons": 10,
            "hidden_activation": "nn.SiLU",
            "output_activation": "nn.Identity",
            "batch_normalization": False,
            "init_weight": {"dist": "uniform", "scale": 1.0e-3},
        }
    )


cs.store(name="poisson1d_meta_solver_default", node=Poisson1DMetaSolverConfig)
