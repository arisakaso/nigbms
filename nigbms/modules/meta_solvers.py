import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module

from nigbms.modules.data import PyTorchLinearSystemTask


class MetaSolver(Module):
    def __init__(self, params_learn: DictConfig, features: DictConfig, model: DictConfig):
        """
        Args:
            params_learn (DictConfig): parameters to learn. key: name of the parameter, value: dimension
            features (DictConfig): input features. key: name of the feature, value: dimension
            model (DictConfig): configuration of base model
        """
        super().__init__()
        self.params_learn = params_learn
        self.features = features
        self.model = model

    def get_mlp_features(self, tau: PyTorchLinearSystemTask) -> Tensor:
        """Arrange input feature for MLP model from Task

        Args:
            tau (Task): Task dataclass

        Returns:
            Tensor: input feature
        """
        features = []
        for k in self.features.keys():
            if k in tau.features:
                features.append(tau.features[k])
        features = torch.cat(features, dim=1).squeeze()  # (bs, dim)
        return features

    def forward(self, tau: PyTorchLinearSystemTask) -> TensorDict:
        """Generate theta (solver parameters) from Task

        Args:
            tau (Task): Task dataclass

        Raises:
            NotImplementedError: _description_

        Returns:
            TensorDict: theta (solver parameters)
        """
        if self.model._get_name() == "MLP":
            x = self.get_mlp_features(tau)
        else:
            raise NotImplementedError(f"Model {self.model._get_name()} not implemented")

        theta = self.model(x)
        return theta


# class MetaSolverMLP(_MetaSolver):
#     def __init__(
#         self,
#         params_learn: DictConfig,
#         features: DictConfig,
#         model: DictConfig,
#     ):
#         super().__init__(params_learn, features, model)

#         if model.layers[0] is None:
#             in_dim = 0
#             for v in features.values():
#                 in_dim += v.dim
#             model.layers[0] = in_dim

#         if model.layers[-1] is None:
#             out_dim = 0
#             for v in params_learn.values():
#                 out_dim += v.dim
#             model.layers[-1] = out_dim

#         self.model = instantiate(model)

#     def forward(self, tau: dict) -> Tensor:
#         bs = tau["A"].shape[0]
#         features = self._make_features(tau)
#         x = torch.cat([features[k].reshape(bs, -1) for k in self.features], dim=-1)
#         theta = self.model(x)
#         return theta


# class MetaSolverUNet2D(_MetaSolver):
#     def __init__(
#         self,
#         params_learn: DictConfig,
#         features: DictConfig,
#         model: DictConfig,
#     ):
#         super().__init__(params_learn, features, model)

#         self.model = instantiate(model)

#     def forward(self, tau: dict) -> Tensor:
#         bs, n2 = tau["b"].shape
#         features = self._make_features(tau)
#         n = int(n2**0.5)
#         x = features["b"].reshape(bs, 1, n, n)
#         theta = self.model(x).reshape(bs, -1)
#         return theta


# # %%
