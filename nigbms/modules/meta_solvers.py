import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module

from nigbms.modules.data import Task


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

    def make_features(self, tau: Task) -> Tensor:
        """Arrange input feature for MLP model from Task

        Args:
            tau (Task): Task dataclass

        Returns:
            Tensor: input features
        """
        features = []
        for k in self.features.keys():
            if k in tau.features:
                features.append(tau.features[k])
        features = torch.cat(features, dim=1).squeeze()  # (bs, dim)
        return features

    def forward(self, tau: Task) -> TensorDict:
        """Generate theta (solver parameters) from Task

        Args:
            tau (Task): Task dataclass

        Raises:
            NotImplementedError: _description_

        Returns:
            TensorDict: theta (solver parameters)
        """
        if self.model._get_name() == "MLP":
            x = self.make_features(tau)
        elif self.model._get_name() == "Constant":
            x = None
        else:
            raise NotImplementedError(f"Model {self.model._get_name()} not implemented")

        theta = self.model(x)
        return theta
