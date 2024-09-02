import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Module

from nigbms.modules.tasks import PETScLinearSystemTask, PyTorchLinearSystemTask, Task
from nigbms.utils.convert import petscvec2tensor


class MetaSolver(Module):
    def __init__(self, params_learn: DictConfig, features: DictConfig, model: Module):
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

    def arrange_input(self, tau: Task) -> Tensor:
        """Arrange input feature for the model from Task

        Args:
            tau (Task): Task dataclass

        Returns:
            Tensor: input features
        """
        raise NotImplementedError

    def forward(self, tau: Task) -> Tensor:
        """Generate theta (solver parameters) from Task

        Args:
            tau (Task): Task dataclass

        Raises:
            NotImplementedError: _description_

        Returns:
            Tensor: theta (solver parameters)
        """
        x = self.arrange_input(tau)
        theta = self.model(x)
        return theta.squeeze()


class Poisson1DMetaSolver(MetaSolver):
    def __init__(self, params_learn: DictConfig, features: DictConfig, model: Module):
        """
        Args:
            params_learn (DictConfig): parameters to learn. key: name of the parameter, value: dimension
            features (DictConfig): input features. key: name of the feature, value: dimension
            model (DictConfig): configuration of base model
        """
        super().__init__(params_learn, features, model)

    def arrange_input(self, tau: PyTorchLinearSystemTask | PETScLinearSystemTask) -> Tensor:
        """Arrange input feature for the model from Task

        Args:
            tau (Task): Task dataclass

        Returns:
            Tensor: input features
        """
        features = []
        for k in self.features.keys():
            if hasattr(tau, k):
                feature = getattr(tau, k)
                if isinstance(tau, PETScLinearSystemTask):
                    feature = list(map(lambda x: petscvec2tensor(x, device=tau.params.device), feature))
                    feature = torch.stack(feature, dim=0)

                features.append(feature)

            elif hasattr(tau.params, k):
                features.append(getattr(tau.params, k))

            else:
                raise ValueError(f"Feature {k} not found in task")

        if "MLP" in self.model.__class__.__name__:
            features = torch.cat(features, dim=1).squeeze(-1)  # (bs, dim)
        else:
            features = torch.stack(features, dim=1).squeeze(-1)  # (bs, channel, dim)

        return features


class ConstantMetaSolver(MetaSolver):
    def __init__(self, params_learn: DictConfig, features: DictConfig, model: Module):
        """
        Args:
            params_learn (DictConfig): parameters to learn. key: name of the parameter, value: dimension
            features (DictConfig): input features. key: name of the feature, value: dimension
            model (DictConfig): configuration of base model
        """
        super().__init__(params_learn, features, model)

    def arrange_input(self, tau: PETScLinearSystemTask) -> Tensor:
        """Arrange input feature for the model from Task

        Args:
            tau (Task): Task dataclass

        Returns:
            Tensor: input features
        """
        return None
