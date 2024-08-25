import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig
from tensordict import TensorDict

from nigbms.configs.surrogates import Poisson1DSurrogateConfig, TestFunctionSurrogateConfig
from nigbms.modules.surrogates import Poisson1DSurrogate, TestFunctionSurrogate
from nigbms.modules.tasks import Task, generate_sample_batched_pytorch_task


class TestPoisson1DSurrogate:
    def test_forward(self):
        with initialize(version_base="1.3"):
            cfg: Poisson1DSurrogateConfig = compose(overrides=["+surrogate@_global_=poisson1d_surrogate_default"])
            cfg.features = DictConfig({"b": [5], "x0": [5]})

        torch.set_default_dtype(torch.float64)
        surrogate = instantiate(cfg)
        assert isinstance(surrogate, Poisson1DSurrogate)

        tau = generate_sample_batched_pytorch_task()
        theta = TensorDict({"x0": torch.zeros_like(tau.b)})
        y = surrogate(tau, theta)
        assert y.shape == torch.Size([len(tau), cfg.model.out_dim])


class TestTestFunctionSurrogate:
    def test_forward(self):
        with initialize(version_base="1.3"):
            cfg: TestFunctionSurrogateConfig = compose(
                overrides=["+surrogate@_global_=testfunction_surrogate_default"]
            )
        surrogate = instantiate(cfg)
        assert isinstance(surrogate, TestFunctionSurrogate)

        batch_size = 10
        tau = Task()
        theta = TensorDict({"x": torch.zeros(batch_size, cfg.features.x[0])})
        y = surrogate(tau, theta)
        assert y.shape == torch.Size([batch_size, cfg.model.out_dim])
