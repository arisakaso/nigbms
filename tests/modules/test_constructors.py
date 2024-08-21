import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from nigbms.configs.constructors import IdentityCodecConfig, SinCodecConfig
from nigbms.modules.constructors import IdentityCodec, SinCodec


def test_identity_codec():
    with initialize(version_base="1.3"):
        cfg: IdentityCodecConfig = compose(overrides=["+codec@_global_=identity_codec_default"])
    codec = instantiate(cfg)
    assert isinstance(codec, IdentityCodec)


class TestSinCodec:
    @pytest.fixture
    def init_codec(self):
        with initialize(version_base="1.3"):
            cfg: SinCodecConfig = compose(overrides=["+codec@_global_=sin_codec_default"])
            self.codec = instantiate(cfg)

    def test_init(self, init_codec):
        assert isinstance(self.codec, SinCodec)
        assert self.codec.basis.shape == (1, self.codec.latent_dim, self.codec.param_dim)

    def test_encode(self, init_codec):
        batch_size = 10
        x = torch.ones(batch_size, self.codec.param_dim)
        z = self.codec.encode(x)
        assert z.shape == (batch_size, self.codec.latent_dim)

    def test_decode(self, init_codec):
        batch_size = 10
        z = torch.ones(batch_size, self.codec.latent_dim)
        x = self.codec.decode(z)
        assert x.shape == (batch_size, self.codec.param_dim)
