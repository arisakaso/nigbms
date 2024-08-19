import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor

from nigbms.modules.data import Task
from nigbms.modules.solvers import _Solver


class SurrogateSolver(_Solver):
    """Surrogate solver class (f_hat)"""

    def __init__(
        self,
        params_fix: DictConfig,
        params_learn: DictConfig,
        features: DictConfig,
        model: DictConfig,
    ) -> None:
        # TODO: Maybe it is better to pass the corresponding solver instead of params_fix and params_learn
        """

        Args:
            params_fix (DictConfig): _description_
            params_learn (DictConfig): _description_
            features (DictConfig): _description_
            model (DictConfig): _description_
        """
        super().__init__(params_fix, params_learn)
        self.features = features
        self.model = model

    def arrange_input(self, tau: Task, theta: TensorDict) -> Tensor:
        raise NotImplementedError

    def forward(self, tau: Task, theta: TensorDict) -> Tensor:
        x = self.arrange_input(tau, theta)
        y = self.model(x)
        y = y.double()
        return y


class Poisson1DSurrogate(SurrogateSolver):
    """Surrogate solver class for the Poisson1D problem"""

    def __init__(
        self,
        params_fix: DictConfig,
        params_learn: DictConfig,
        features: DictConfig,
        model: DictConfig,
    ) -> None:
        super().__init__(params_fix, params_learn, features, model)

    def arrange_input(self, tau: Task, theta: TensorDict) -> Tensor:
        features = []
        for k in self.features.keys():
            if hasattr(tau, k):
                features.append(getattr(tau, k))
            elif hasattr(tau.params, k):
                features.append(getattr(tau.params, k))
            elif k in theta:
                features.append(theta[k])
            else:
                raise ValueError(f"Feature {k} not found in task")

        features = torch.cat(features, dim=1).squeeze()  # (bs, dim)
        features = features.float()  # neural network expects floats
        return features


class TestFunctionSurrogate(SurrogateSolver):
    """Surrogate solver class for the Poisson1D problem"""

    def __init__(
        self,
        params_fix: DictConfig,
        params_learn: DictConfig,
        features: DictConfig,
        model: DictConfig,
    ) -> None:
        super().__init__(params_fix, params_learn, features, model)

    def arrange_input(self, tau: Task, theta: TensorDict) -> Tensor:
        features = []
        for k in self.features.keys():
            if hasattr(tau, k):
                features.append(getattr(tau, k))
            elif hasattr(tau.params, k):
                features.append(getattr(tau.params, k))
            elif k in theta:
                features.append(theta[k])
            else:
                raise ValueError(f"Feature {k} not found in task")

        features = torch.cat(features, dim=1).squeeze()  # (bs, dim)
        # features = features.float()  # neural network expects floats
        return features


# import torch
# from omegaconf import DictConfig
# from torch import Tensor

# from nigbms.modules.data import PyTorchLinearSystemTask
# from nigbms.modules.solvers import _Solver


# class SurrogateSolver(_Solver):
#     """Surrogate solver class (f_hat)"""

#     def __init__(
#         self,
#         params_fix: DictConfig,
#         params_learn: DictConfig,
#         features: DictConfig,
#         model: DictConfig,
#     ) -> None:
#         # TODO: Maybe it is better to pass the corresponding solver instead of params_fix and params_learn
#         """

#         Args:
#             params_fix (DictConfig): _description_
#             params_learn (DictConfig): _description_
#             features (DictConfig): _description_
#             model (DictConfig): _description_
#         """
#         super().__init__(params_fix, params_learn)
#         self.features = features
#         self.model = model

#     def get_mlp_features(self, tau: PyTorchLinearSystemTask, theta: Tensor) -> Tensor:
#         features = []
#         for k in self.features.keys():
#             if k in theta:
#                 features.append(theta[k])
#             elif k in tau.features:
#                 features.append(tau.features[k])
#         features = torch.cat(features, dim=1).squeeze()  # (bs, dim)
#         return features

#     def get_conv_features(self, tau: PyTorchLinearSystemTask, theta: Tensor) -> Tensor:
#         features = []
#         for k in self.features.keys():
#             if k in theta:
#                 features.append(theta[k])
#             elif k in tau:
#                 features.append(tau[k])
#         features = torch.cat(features, dim=1).unsqueeze(1)  # (bs, dim)

#         return features

#     def forward(self, tau: PyTorchLinearSystemTask, theta: Tensor) -> Tensor:
#         if self.model._get_name() == "MLP":
#             x = self.get_mlp_features(tau, theta)
#         elif self.model._get_name() == "CNN1D":
#             x = self.get_conv_features(tau, theta)
#         else:
#             raise NotImplementedError(f"Model {self.model._get_name()} not implemented")

#         y = self.model(x)
#         return y
