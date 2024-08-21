from dataclasses import dataclass

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


@dataclass
class IdentityCodecConfig:
    """IdentityCodecConfig class."""

    _target_: str = "nigbms.modules.constructors.IdentityCodec"
    param_dim: int = 128
    latent_dim: int = 128


cs.store(name="identity_codec_default", group="codec", node=IdentityCodecConfig)


@dataclass
class SinCodecConfig:
    """SinCodecConfig class."""

    _target_: str = "nigbms.modules.constructors.SinCodec"
    param_dim: int = 128
    latent_dim: int = 64


cs.store(name="sin_codec_default", group="codec", node=SinCodecConfig)
