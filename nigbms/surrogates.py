import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.functional import relu, sigmoid  # noqa

from nigbms.constructors import FFTCodec
from nigbms.solvers import AbstractSolver
from nigbms.tasks import PETScLinearSystemTask, Task
from nigbms.utils.convert import petscvec2tensor


class SurrogateSolver(AbstractSolver):
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

            elif k == "e0_fft":
                x = tau.x
                x0 = theta["x0"]
                codec = FFTCodec(param_dim=31, latent_dim=32)
                e0_fft = codec.encode(x - x0).unsqueeze(-1)
                features.append(e0_fft)

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


class ExponentialDecaySurrogate(SurrogateSolver):
    """Surrogate solver class for the Poisson1D problem"""

    def __init__(
        self,
        params_fix: DictConfig,
        params_learn: DictConfig,
        features: DictConfig,
        model: Module,
        constructor: Module = None,
        n_components: int = 31,
    ) -> None:
        super().__init__(params_fix, params_learn, features, model, constructor)
        # self.decay_rates = Parameter(torch.rand(n_components, 1))  # (n_components, 1)
        self.decay_rates = Parameter(torch.randn(n_components, 1) * 10)  # (n_components, 1)
        self.roll_out = Parameter(
            torch.arange(params_fix.history_length).unsqueeze(0), requires_grad=False
        )  # (1, out_dim)

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

            elif k == "e0_fft":
                x = tau.x
                x0 = theta["x0"]
                codec = FFTCodec(param_dim=31, latent_dim=32)
                e0_fft = codec.encode(x - x0).unsqueeze(-1)
                features.append(e0_fft)

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

        if "MLP" in self.model.__class__.__name__:
            features = torch.cat(features, dim=1).squeeze()  # (bs, dim)
        elif "CNN" in self.model.__class__.__name__:
            features = torch.stack(features, dim=1).squeeze()  # (bs, channel, dim)
        else:
            raise ValueError("Model type not supported")
        return features

    def forward(self, tau: Task, theta: TensorDict) -> Tensor:
        x = self.arrange_input(tau, theta)
        c = self.model(x)  # (bs, n_components)

        base = torch.pow(sigmoid(self.decay_rates), self.roll_out)  # (n_components, out_dim)
        y = c @ base  # (bs, out_dim)
        return y


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

        if "MLP" in self.model.__class__.__name__:
            features = torch.cat(features, dim=1).squeeze()  # (bs, dim)
        elif "CNN" in self.model.__class__.__name__:
            features = torch.stack(features, dim=1)  # (bs, channel, dim)
        else:
            raise ValueError("Model type not supported")

        return features
