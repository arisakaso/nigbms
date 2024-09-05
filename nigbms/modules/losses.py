import torch
from torch import Tensor, log, norm, sigmoid
from torch.nn import Module
from torch.nn.functional import mse_loss

from nigbms.modules.data import PyTorchLinearSystemTask
from nigbms.modules.tasks import PETScLinearSystemTask
from nigbms.utils.convert import petscvec2tensor


class SurrogateSolverLoss(Module):
    """Loss function for training surrogate solvers (f_hat)"""

    def __init__(self, weights: dict, reduce: bool, mask: bool = False, conservative: bool = False) -> None:
        """

        Args:
            weights (dict): weights for each loss term
            reduce (bool): whether to reduce the loss to a scalar
        """
        super().__init__()
        self.weights = weights
        self.reduce = reduce
        self.mask = mask
        self.conservative = conservative

    def forward(self, y, y_hat, dvf, dvf_hat, dvL, dvL_hat) -> dict:
        """Compute the loss

        Args:
            y (_type_): output of the solver i.e. history of residuals
            y_hat (_type_): output of the surrogate solver
            dvf (_type_): directional derivative of the solver
            dvf_hat (_type_): directional derivative of the surrogate solver
            dvL (_type_): dL/dy * dvf
            dvL_hat (_type_): dL/dy * dvf_hat

        Returns:
            dict: dict of losses
        """
        is_converged = y == 0

        if self.conservative:
            last_false_inds = (~is_converged).sum(dim=1) - 1
            batch_inds = torch.arange(is_converged.size(0))
            is_converged[batch_inds, last_false_inds] = True

        if self.mask:
            y_hat = torch.where(is_converged, 0, y_hat)
            dvf_hat = torch.where(is_converged, 0, dvf_hat)
            dvf = torch.where(is_converged, 0, dvf)

        losses = {
            "y_loss": mse_loss(y, y_hat),
            "y_loss_relative": (torch.where(is_converged, 0, (y - y_hat) / y) ** 2).mean(),
            "dvf_loss": mse_loss(dvf, dvf_hat),
            "dvf_loss_relative": (torch.where(is_converged, 0, (dvf - dvf_hat) / dvf) ** 2).mean(),
            "dvL_loss": mse_loss(dvL, dvL_hat),
            "loss": 0,
            "dvf_max": dvf.max(),  # for debugging
            "dvf_min": dvf.min(),  # for debugging
        }

        for k, w in self.weights.items():
            losses["loss"] += w * losses[k]

        # if self.reduce:
        #     losses = {k: v.mean() for k, v in losses.items()}

        return losses


class MetaSolverLoss(Module):
    """Loss function for training meta solvers (f)"""

    def __init__(self, weights: dict, reduce: bool, constructor, gain: float = 1.0) -> None:
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
        self.gain = gain

    def forward(self, tau: PyTorchLinearSystemTask | PETScLinearSystemTask, theta: Tensor, history: Tensor) -> dict:
        """Compute the loss

        Args:
            tau (Task): Task dataclass
            theta (Tensor): solver parameters
            history (Tensor): convergence history (residuals)

        Returns:
            dict: dict of losses
        """
        theta = self.constructor(theta)

        if isinstance(tau, PyTorchLinearSystemTask):
            bnorm = norm(tau.b, dim=(1, 2))  # (bs,)
            xnorm = norm(tau.x, dim=(1, 2))  # (bs,)
            x = tau.x  # (bs, n, 1)
        elif isinstance(tau, PETScLinearSystemTask):
            bnorm = torch.tensor([b.norm() for b in tau.b], device=theta.device)
            xnorm = torch.tensor([x.norm() for x in tau.x], device=theta.device)
            x = list(map(lambda x: petscvec2tensor(x, device=theta.device), tau.x))
            x = torch.stack(x, dim=0)  # (bs, n, 1)
        else:  # pragma: no cover
            raise ValueError(f"Task type {type(tau)} not supported")

        rtol = tau.rtol.unsqueeze(-1)  # (bs, 1)
        bnorm = bnorm.unsqueeze(-1)  # (bs, 1)
        relative_history = history / bnorm  # (bs, niter)
        x0 = theta["x0"]
        r0 = history[:, 0]
        relative_r0 = relative_history[:, 0]
        # TODO: This r0 is dependent on the solver, and the backward pass can be modified.
        # It can be computed as a function of x0 without involving the solver.
        e0 = norm(x - x0, dim=(1, 2))

        is_converged = relative_history < rtol

        loss_dict = {
            # total loss
            "loss": 0,
            # solver independent
            "r0": r0,
            "relative_r0": relative_r0,
            "e0": e0,
            "relative_e0": e0 / xnorm,
            # solver dependent
            "iter_r": (~is_converged).sum(dim=1).float(),
            "iter_r_proxy": torch.where(is_converged, 0, sigmoid(self.gain * (relative_history - rtol))).sum(dim=1),
            "iter_r_proxy_log": torch.where(is_converged, 0, sigmoid(log(relative_history / rtol))).sum(dim=1),
            "rn": history[:, -1],
            "relative_rn": relative_history[:, -1],
        }
        for k, w in self.weights.items():
            loss_dict["loss"] += w * loss_dict[k]

        if self.reduce:
            loss_dict = {k: v.mean() for k, v in loss_dict.items()}

        return loss_dict
