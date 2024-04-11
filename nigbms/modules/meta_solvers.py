import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module


class _MetaSolver(Module):
    def __init__(self, params_learn: DictConfig, features: DictConfig, model: DictConfig):
        super().__init__()
        self.params_learn = params_learn
        self.features = features
        self.model = model

    def _make_features(self, tau: dict) -> Tensor:
        raise NotImplementedError

    def forward(self, tau: dict) -> Tensor:
        raise NotImplementedError


class MetaSolverMLP(_MetaSolver):
    def __init__(
        self,
        params_learn: DictConfig,
        features: DictConfig,
        model: DictConfig,
    ):
        super().__init__(params_learn, features, model)

        if model.layers[0] is None:
            in_dim = 0
            for v in features.values():
                in_dim += v.dim
            model.layers[0] = in_dim

        if model.layers[-1] is None:
            out_dim = 0
            for v in params_learn.values():
                out_dim += v.dim
            model.layers[-1] = out_dim

        self.model = instantiate(model)

    def forward(self, tau: dict) -> Tensor:
        bs = tau["A"].shape[0]
        features = self._make_features(tau)
        x = torch.cat([features[k].reshape(bs, -1) for k in self.features], dim=-1)
        theta = self.model(x)
        return theta


class MetaSolverUNet2D(_MetaSolver):
    def __init__(
        self,
        params_learn: DictConfig,
        features: DictConfig,
        model: DictConfig,
    ):
        super().__init__(params_learn, features, model)

        self.model = instantiate(model)

    def forward(self, tau: dict) -> Tensor:
        bs, n2 = tau["b"].shape
        features = self._make_features(tau)
        n = int(n2**0.5)
        x = features["b"].reshape(bs, 1, n, n)
        theta = self.model(x).reshape(bs, -1)
        return theta


# %%
