from dataclasses import dataclass

from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

cs = ConfigStore.instance()


@dataclass
class IdentityCodecConfig:
    """IdentityCodecConfig class."""

    _target_: str = "nigbms.constructors.IdentityCodec"
    param_dim: int = 128
    latent_dim: int = 128


cs.store(name="identity_codec_default", group="codec", node=IdentityCodecConfig)


@dataclass
class SinCodecConfig:
    """SinCodecConfig class."""

    _target_: str = "nigbms.constructors.SinCodec"
    param_dim: int = 128
    latent_dim: int = 64


cs.store(name="sin_codec_default", group="codec", node=SinCodecConfig)


@dataclass
class LinearCodecConfig:
    """SinCodecConfig class."""

    _target_: str = "nigbms.constructors.LinearCodec"
    param_dim: int = 128
    latent_dim: int = 128


cs.store(name="linear_codec_default", group="codec", node=LinearCodecConfig)


@dataclass
class FFTCodecConfig:
    """FFTCodecConfig class."""

    _target_: str = "nigbms.constructors.FFTCodec"
    param_dim: int = 128
    latent_dim: int = 128


cs.store(name="fft_codec_default", group="codec", node=FFTCodecConfig)


@dataclass
class ConstructorConfig:
    """ConstructorConfig class."""

    _target_: str = "nigbms.constructors.ThetaConstructor"
    params: DictConfig = DictConfig(
        {
            "x": DictConfig(
                {
                    "codec": IdentityCodecConfig(param_dim=10, latent_dim=10),
                    "shape": [5, 2],
                }
            ),
            "y": DictConfig(
                {
                    "codec": SinCodecConfig(param_dim=32, latent_dim=16),
                    "shape": [32],
                }
            ),
        }
    )


cs.store(name="constructor_default", group="constructor", node=ConstructorConfig)


@dataclass
class ConstructorTestFunctionConfig:
    """ConstructorConfig class."""

    _target_: str = "nigbms.constructors.ThetaConstructor"
    params: DictConfig = DictConfig(
        {
            "x": DictConfig(
                {
                    "codec": IdentityCodecConfig(param_dim=10, latent_dim=10),
                    "shape": [5, 2],
                }
            )
        }
    )


cs.store(name="constructor_testfunction_default", group="constructor", node=ConstructorTestFunctionConfig)
