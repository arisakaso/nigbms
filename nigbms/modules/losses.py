import torch
from torch import Tensor, log, norm, sigmoid
from torch.nn import Module
from torch.nn.functional import mse_loss

from nigbms.modules.data import PyTorchLinearSystemTask


class SurrogateSolverLoss(Module):
    """Loss function for training surrogate solvers (f_hat)"""

    def __init__(self, weights: dict, reduce: bool) -> None:
        """

        Args:
            weights (dict): weights for each loss term
            reduce (bool): whether to reduce the loss to a scalar
        """
        super().__init__()
        self.weights = weights
        self.reduce = reduce

    def forward(self, y, y_hat, dvf, dvf_hat, dvL, dvL_hat) -> dict:
        """Compute the loss

        Args:
            y (_type_): output of the solver
            y_hat (_type_): output of the surrogate solver
            dvf (_type_): directional derivative of the solver
            dvf_hat (_type_): directional derivative of the surrogate solver
            dvL (_type_): dL/dy * dvf
            dvL_hat (_type_): dL/dy * dvf_hat

        Returns:
            dict: dict of losses
        """

        losses = {
            "y_loss": mse_loss(y, y_hat),
            "dvf_loss": mse_loss(dvf, dvf_hat),
            "dvf_loss_relative": (((dvf - dvf_hat) / dvf) ** 2),
            "dvL_loss": mse_loss(dvL, dvL_hat),
            "loss": 0,
        }

        for k, w in self.weights.items():
            losses["loss"] += w * losses[k]

        if self.reduce:
            losses = {k: v.mean() for k, v in losses.items()}

        return losses


class MetaSolverLoss(Module):
    """Loss function for training meta solvers (f)"""

    def __init__(self, weights: dict, reduce: bool, constructor) -> None:
        """

        Args:
            weights (dict): weights for each loss term
            reduce (bool): whether to reduce the loss to a scalar
            constructor (_type_): theta constructor
        """
        super().__init__()
        self.weights = weights
        self.reduce = reduce
        self.constructor = constructor

    def forward(self, tau: PyTorchLinearSystemTask, theta: Tensor, history: Tensor) -> dict:
        """Compute the loss

        Args:
            tau (Task): Task dataclass
            theta (Tensor): solver parameters
            history (Tensor): convergence history (residuals)

        Returns:
            dict: dict of losses
        """
        theta = self.constructor(theta)

        bnorm = norm(tau.b, dim=(1, 2))  # (bs,)
        xnorm = norm(tau.x, dim=(1, 2))  # (bs,)
        rtol_bnorm = (tau.rtol * bnorm).unsqueeze(-1)  # (bs, 1)
        x0 = theta["x0"]  # (bs, n, 1)
        r0 = norm(tau.b - tau.A @ x0, dim=(1, 2))  # (bs,)
        e0 = norm(tau.x - x0, dim=(1, 2))  # (bs,)
        unconvergece_flag = history > rtol_bnorm

        loss_dict = {
            # total loss
            "loss": torch.zeros_like(bnorm),
            # solver independent
            "r0": r0,
            "r0^2": r0**2,
            "relative_r0": r0 / bnorm,
            "relative_r0^2": (r0 / bnorm) ** 2,
            "e0": e0,
            "e0^2": e0**2,
            "relative_e0": e0 / xnorm,
            "relative_e0^2": (e0 / xnorm) ** 2,
            # solver dependent
            "iter_r": unconvergece_flag.sum(dim=1).float(),
            "iter_r_proxy": torch.where(unconvergece_flag, sigmoid(history - rtol_bnorm), 0).sum(dim=1),
            "iter_r_proxy_log": torch.where(unconvergece_flag, sigmoid(log(history / rtol_bnorm)), 0).sum(dim=1),
            "rn": history[:, -1],
            "rn^2": history[:, -1] ** 2,
            "relative_rn": history[:, -1] / bnorm,
        }
        for k, w in self.weights.items():
            loss_dict["loss"] += w * loss_dict[k]

        if self.reduce:
            loss_dict = {k: v.mean() for k, v in loss_dict.items()}

        return loss_dict
