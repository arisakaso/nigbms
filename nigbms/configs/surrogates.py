from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from nigbms.utils.resolver import calc_in_dim

OmegaConf.register_new_resolver("calc_in_dim", calc_in_dim)
cs = ConfigStore.instance()


@dataclass
class Poisson1DSurrogateConfig:
    """Poisson1DSurrogateConfig class."""

    _target_: str = "nigbms.modules.surrogates.Poisson1DSurrogate"
    params_fix: DictConfig = DictConfig({})
    params_learn: DictConfig = DictConfig({"x0": [5]})
    features: DictConfig = DictConfig({"b": [5], "x0": [5]})
    model: DictConfig = DictConfig(
        {
            "_target_": "nigbms.modules.models.MLP",
            "in_dim": "${calc_in_dim:${..features}}",
            "out_dim": 100,
            "num_layers": 3,
            "num_neurons": 10,
            "hidden_activation": {"_target_": "torch.nn.SiLU"},
            "output_activation": {"_target_": "torch.nn.Identity"},
            "batch_normalization": False,
            "init_weight": {"dist": "uniform", "scale": 1.0e-3},
        }
    )


cs.store(name="poisson1d_surrogate_default", group="surrogate", node=Poisson1DSurrogateConfig)


@dataclass
class TestFunctionSurrogateConfig:
    _target_: str = "nigbms.modules.surrogates.TestFunctionSurrogate"
    params_fix: DictConfig = DictConfig({})
    params_learn: DictConfig = DictConfig({"x": [5]})
    features: DictConfig = DictConfig({"x": [5]})
    model: DictConfig = DictConfig(
        {
            "_target_": "nigbms.modules.models.MLP",
            "in_dim": "${calc_in_dim:${..features}}",
            "out_dim": 1,
            "num_layers": 1,
            "num_neurons": 1024,
            "hidden_activation": {"_target_": "torch.nn.GELU"},
            "output_activation": {"_target_": "torch.nn.Identity"},
            "batch_normalization": False,
            "init_weight": {"dist": "normal", "scale": 1.0e-1},
        }
    )


cs.store(name="testfunction_surrogate_default", group="surrogate", node=TestFunctionSurrogateConfig)
