import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from nigbms.configs.constructors import IdentityCodecConfig, SinCodecConfig
from nigbms.modules.constructors import IdentityCodec, SinCodec, ThetaConstructor


class TestIdentityCodec:
    @pytest.fixture
    def init_codec(self):
        with initialize(version_base="1.3"):
            cfg: IdentityCodecConfig = compose(overrides=["+codec@_global_=identity_codec_default"])
            self.codec = instantiate(cfg)

    def test_init(self, init_codec):
        assert isinstance(self.codec, IdentityCodec)
        assert self.codec.param_dim == self.codec.latent_dim

    def test_encode(self, init_codec):
        batch_size = 10
        x = torch.ones(batch_size, self.codec.param_dim)
        z = self.codec.encode(x)
        assert torch.equal(z, x)

    def test_decode(self, init_codec):
        batch_size = 10
        z = torch.ones(batch_size, self.codec.latent_dim)
        x = self.codec.decode(z)
        assert torch.equal(x, z)


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


class TestConstructor:
    @pytest.fixture
    def init_constructor(self):
        with initialize(version_base="1.3"):
            cfg = compose(overrides=["+constructor@_global_=constructor_default"])
            self.constructor = instantiate(cfg)

    def test_init(self, init_constructor):
        assert isinstance(self.constructor, ThetaConstructor)

    def test_forward(self, init_constructor):
        batch_size = 10
        x = self.constructor.params.x
        y = self.constructor.params.y
        theta = torch.randn(batch_size, x.codec.param_dim + y.codec.param_dim)
        theta_dict = self.constructor(theta)
        assert theta_dict["x"].shape == (batch_size, *x.shape)
        assert theta_dict["y"].shape == (batch_size, *y.shape)
