import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module

from nigbms.modules.solvers import _Solver
from nigbms.modules.tasks import PETScLinearSystemTask, Task
from nigbms.utils.convert import petscvec2tensor


class SurrogateSolver(_Solver):
    """Surrogate solver class (f_hat)"""

    def __init__(
        self,
        params_fix: DictConfig,
        params_learn: DictConfig,
        features: DictConfig,
        model: Module,
        constructor: Module = None,
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
        self.constructor = constructor

    def arrange_input(self, tau: Task, theta: TensorDict) -> Tensor:
        raise NotImplementedError

    def forward(self, tau: Task, theta: TensorDict) -> Tensor:
        x = self.arrange_input(tau, theta)
        y = self.model(x)
        return y


class Poisson1DSurrogate(SurrogateSolver):
    """Surrogate solver class for the Poisson1D problem"""

    def __init__(
        self,
        params_fix: DictConfig,
        params_learn: DictConfig,
        features: DictConfig,
        model: Module,
        constructor: Module = None,
    ) -> None:
        super().__init__(params_fix, params_learn, features, model, constructor)

    def arrange_input(self, tau: Task, theta: TensorDict) -> Tensor:
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

            elif k in theta:
                features.append(theta[k])

            elif k == "e0":
                if isinstance(tau, PETScLinearSystemTask):
                    x = list(map(lambda x: petscvec2tensor(x, device=tau.params.device), tau.x))
                    x = torch.stack(x, dim=0)
                else:
                    x = tau.x

                x0 = theta["x0"]
                features.append(x - x0)

            elif k == "x0_enc":
                x0_enc = self.constructor.params.x0.codec.encode(theta["x0"]).unsqueeze(-1)
                features.append(x0_enc)

            elif k == "e0_enc":
                if isinstance(tau, PETScLinearSystemTask):
                    x = list(map(lambda x: petscvec2tensor(x, device=tau.params.device), tau.x))
                    x = torch.stack(x, dim=0)
                else:
                    x = tau.x
                x_enc = self.constructor.params.x0.codec.encode(x).unsqueeze(-1)
                x0_enc = self.constructor.params.x0.codec.encode(theta["x0"]).unsqueeze(-1)
                features.append((x0_enc - x_enc).abs())

            else:
                raise ValueError(f"Feature {k} not found in task")

        features = torch.cat(features, dim=1).squeeze()  # (bs, dim)
        return features


class Poisson1DSurrogate(SurrogateSolver):
    """Surrogate solver class for the Poisson1D problem"""

    def __init__(
        self,
        params_fix: DictConfig,
        params_learn: DictConfig,
        features: DictConfig,
        model: Module,
        constructor: Module = None,
    ) -> None:
        super().__init__(params_fix, params_learn, features, model, constructor)

    def arrange_input(self, tau: Task, theta: TensorDict) -> Tensor:
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

            elif k in theta:
                features.append(theta[k])

            elif k == "e0":
                if isinstance(tau, PETScLinearSystemTask):
                    x = list(map(lambda x: petscvec2tensor(x, device=tau.params.device), tau.x))
                    x = torch.stack(x, dim=0)
                else:
                    x = tau.x

                x0 = theta["x0"]
                features.append(x - x0)

            elif k == "x0_enc":
                x0_enc = self.constructor.params.x0.codec.encode(theta["x0"]).unsqueeze(-1)
                features.append(x0_enc)

            elif k == "e0_enc":
                if isinstance(tau, PETScLinearSystemTask):
                    x = list(map(lambda x: petscvec2tensor(x, device=tau.params.device), tau.x))
                    x = torch.stack(x, dim=0)
                else:
                    x = tau.x
                x_enc = self.constructor.params.x0.codec.encode(x).unsqueeze(-1)
                x0_enc = self.constructor.params.x0.codec.encode(theta["x0"]).unsqueeze(-1)
                features.append((x0_enc - x_enc).abs())

            else:
                raise ValueError(f"Feature {k} not found in task")

        features = torch.cat(features, dim=1).squeeze()  # (bs, dim)
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
