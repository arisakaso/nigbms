from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from nigbms.utils.resolver import calc_in_channels, calc_in_dim
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("calc_in_dim", calc_in_dim)
OmegaConf.register_new_resolver("calc_in_channels", calc_in_channels)
cs = ConfigStore.instance()


@dataclass
class Poisson1DSurrogateConfig:
    """Poisson1DSurrogateConfig class."""

    _target_: str = "nigbms.modules.surrogates.Poisson1DSurrogate"
    params_fix: DictConfig = DictConfig({})
    params_learn: DictConfig = DictConfig({"x0": [5]})
    features: DictConfig = DictConfig({})
    model: DictConfig = DictConfig(
        {
            "_target_": "nigbms.modules.models.MLP",
            "in_dim": "${calc_in_dim:${..features}}",
            "out_dim": 100,
            "n_layers": 3,
            "n_units": 10,
            "hidden_activation": {"_target_": "torch.nn.SiLU"},
            "output_activation": {"_target_": "torch.nn.Identity"},
            "batch_normalization": False,
            "init_weight": {"dist": "uniform", "scale": 1.0e-3},
        }
    )


cs.store(name="poisson1d_surrogate_default", group="surrogate", node=Poisson1DSurrogateConfig)


@dataclass
class ExponentialDecaySurrogateConfig:
    """Poisson1DSurrogateConfig class."""

    _target_: str = "nigbms.modules.surrogates.ExponentialDecaySurrogate"
    params_fix: DictConfig = DictConfig({})
    params_learn: DictConfig = DictConfig({"x0": [5]})
    features: DictConfig = DictConfig({})
    model: DictConfig = DictConfig(
        {
            "_target_": "nigbms.modules.models.MLP",
            "in_dim": "${calc_in_dim:${..features}}",
            "in_channels": "${calc_in_channels:${..features}}",
            "out_dim": "${..n_components}",
            "n_layers": 3,
            "n_units": 10,
            "hidden_activation": {"_target_": "torch.nn.SiLU"},
            "output_activation": {"_target_": "torch.nn.Identity"},
            "batch_normalization": False,
            "init_weight": {"dist": "uniform", "scale": 1.0e-3},
        }
    )
    n_components: int = 10


cs.store(name="exponential_decay_default", group="surrogate", node=ExponentialDecaySurrogateConfig)


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
            "n_layers": 1,
            "n_units": 1024,
            "hidden_activation": {"_target_": "torch.nn.GELU"},
            "output_activation": {"_target_": "torch.nn.Identity"},
            "batch_normalization": False,
            "init_weight": {"dist": "normal", "scale": 1.0e-1},
        }
    )


cs.store(name="testfunction_surrogate_default", group="surrogate", node=TestFunctionSurrogateConfig)
