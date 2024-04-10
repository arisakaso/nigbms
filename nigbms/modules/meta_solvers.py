# %%
from typing import Dict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module

from nigbms import FFTEncoder, SinEncoder


class MetaSolver(Module):
    def __init__(self, params_learn: DictConfig, features: DictConfig, model: DictConfig):
        super().__init__()
        self.params_learn = params_learn
        self.features = features
        self.model = model

    def _get_features(self, tau: Dict):
        features = {}

        if "b" in self.features:
            features["b"] = tau["b"].unsqueeze(-1)

        if "b_freq" in self.features:
            features["b_freq"] = FFTEncoder(self.features.b_freq.dim)(tau["b"])

        if "b_sin" in self.features:
            features["b_sin"] = SinEncoder(self.features.b_sin.dim)(tau["b"])

        if "features" in self.features:
            features["features"] = torch.log(tau["features"])

        return features

    def forward(self, tau: Dict) -> Tensor:
        raise NotImplementedError


class MetaSolverMLP(MetaSolver):
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

    def forward(self, tau: Dict) -> Tensor:
        bs = tau["A"].shape[0]
        features = self._get_features(tau)
        x = torch.cat([features[k].reshape(bs, -1) for k in self.features], dim=-1)
        theta = self.model(x)
        return theta


class MetaSolverUNet2D(MetaSolver):
    def __init__(
        self,
        params_learn: DictConfig,
        features: DictConfig,
        model: DictConfig,
    ):
        super().__init__(params_learn, features, model)

        self.model = instantiate(model)

    def forward(self, tau: Dict) -> Tensor:
        bs, n2 = tau["b"].shape
        features = self._get_features(tau)
        n = int(n2**0.5)
        x = features["b"].reshape(bs, 1, n, n)
        theta = self.model(x).reshape(bs, -1)
        return theta


# %%
