import torch
from omegaconf import DictConfig
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

    def get_mlp_features(self, tau: Task, theta: Tensor) -> Tensor:
        features = []
        for k in self.features.keys():
            if k in theta:
                features.append(theta[k])
            elif k in tau.features:
                features.append(tau.features[k])
        features = torch.cat(features, dim=1).squeeze()  # (bs, dim)
        return features

    def get_conv_features(self, tau: Task, theta: Tensor) -> Tensor:
        features = []
        for k in self.features.keys():
            if k in theta:
                features.append(theta[k])
            elif k in tau:
                features.append(tau[k])
        features = torch.cat(features, dim=1).unsqueeze(1)  # (bs, dim)

        return features

    def forward(self, tau: Task, theta: Tensor) -> Tensor:
        if self.model._get_name() == "MLP":
            x = self.get_mlp_features(tau, theta)
        elif self.model._get_name() == "CNN1D":
            x = self.get_conv_features(tau, theta)
        else:
            raise NotImplementedError(f"Model {self.model._get_name()} not implemented")

        y = self.model(x)
        return y
