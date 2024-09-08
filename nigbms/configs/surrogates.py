from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from nigbms.utils.resolver import calc_in_channels, calc_in_dim
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("calc_in_dim", calc_in_dim)
OmegaConf.register_new_resolver("calc_in_channels", calc_in_channels)
cs = ConfigStore.instance()


@dataclass
class SurrogateSolverConfig:
    """SurrogateSolverConfig class."""

    _target_: str = "nigbms.modules.surrogates.SurrogateSolver"
    params_fix: DictConfig = DictConfig({})
    params_learn: DictConfig = DictConfig({})
    features: DictConfig = DictConfig({})
    model: DictConfig = DictConfig({})


@dataclass
class Poisson1DSurrogateConfig(SurrogateSolverConfig):
    """Poisson1DSurrogateConfig class."""

    _target_: str = "nigbms.modules.surrogates.Poisson1DSurrogate"
    model: DictConfig = DictConfig(
        {
            "_target_": "nigbms.modules.models.MLP",
            "in_dim": "${calc_in_dim:${..features}}",
        }
    )


cs.store(name="poisson1d_surrogate_default", group="surrogate", node=Poisson1DSurrogateConfig)


@dataclass
class ExponentialDecaySurrogateConfig(SurrogateSolverConfig):
    """Poisson1DSurrogateConfig class."""

    _target_: str = "nigbms.modules.surrogates.ExponentialDecaySurrogate"
    model: DictConfig = DictConfig(
        {
            "_target_": "nigbms.modules.models.MLP",
            "in_dim": "${calc_in_dim:${..features}}",
            "in_channels": "${calc_in_channels:${..features}}",
            "out_dim": "${..n_components}",
        }
    )
    n_components: int = 10


cs.store(name="exponential_decay_default", group="surrogate", node=ExponentialDecaySurrogateConfig)


@dataclass
class TestFunctionSurrogateConfig(SurrogateSolverConfig):
    _target_: str = "nigbms.modules.surrogates.TestFunctionSurrogate"
    params_learn: DictConfig = DictConfig({"x": [5]})
    features: DictConfig = DictConfig({"x": [5]})
    model: DictConfig = DictConfig(
        {
            "_target_": "nigbms.modules.models.MLP",
            "in_dim": "${calc_in_dim:${..features}}",
            "in_channels": "${calc_in_channels:${..features}}",
            "out_dim": 1,
        }
    )


cs.store(name="testfunction_surrogate_default", group="surrogate", node=TestFunctionSurrogateConfig)
