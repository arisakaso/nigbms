# %%
from typing import Any, Dict

import torch
from torch import nn, norm
from torch.nn.functional import mse_loss


class SolverIndependentLoss(nn.Module):
    def __init__(self, weights: Dict[str, Any], reduce=False) -> None:
        super().__init__()
        self.weights = weights
        self.reduce = reduce

    def forward(self, tau, theta):
        rtol = tau["rtol"]  # (bs,)
        bnorm = norm(tau["b"], dim=1)  # (bs,)
        xnorm = norm(tau["x_sol"], dim=1)  # (bs,)
        tol_bnorm = rtol * bnorm
        tol_xnorm = rtol * xnorm
        x0 = theta.unsqueeze(-1)  # (bs, n, 1)
        A = tau["A"]
        b = tau["b"].unsqueeze(-1)  # (bs, n, 1)
        x_sol = tau["x_sol"].unsqueeze(-1)  # (bs, n, 1)

        r0 = norm(b - torch.bmm(A, x0), dim=(1, 2))  # (bs,)
        e0 = norm(x_sol - x0, dim=(1, 2))  # (bs,)
        losses = {
            "initial_residual": r0,
            "initial_residual_relative": r0 / bnorm,
            "initial_residual_relative_squared": r0**2 / bnorm**2,
            "number_of_iterations_residual_surrogate": torch.where(
                r0 > tol_bnorm, torch.sigmoid(r0 - tol_bnorm), 0
            ),
            "initial_error": e0,
            "initial_error_relative": e0 / xnorm,
            "initial_error_relative_squared": e0**2 / xnorm**2,
            "number_of_iterations_error_surrogate": torch.where(
                e0 > tol_xnorm, torch.sigmoid(r0 - tol_xnorm), 0
            ),
            "number_of_iterations_error_surrogate_log": torch.where(
                e0 > tol_xnorm, torch.sigmoid(torch.log(e0) - torch.log(tol_xnorm)), 0
            ),
        }
        for k, w in self.weights.items():
            losses["i_loss"] = w * losses[k]

        return losses


class SolverDependentLoss(nn.Module):
    def __init__(self, weights: Dict[str, Any], reduce=False) -> None:
        super().__init__()
        self.weights = weights
        self.reduce = reduce

    def forward(self, tau, history):
        rtol = tau["rtol"].unsqueeze(-1)  # (bs,)

        losses = {"d_loss": torch.zeros_like(rtol)}

        losses["number_of_iterations"] = (history > rtol).sum(dim=1).float()
        losses["number_of_iterations_surrogate"] = torch.where(
            history > rtol, torch.sigmoid(history - rtol), 0
        ).sum(dim=1)
        losses["last_step_relative"] = history[:, -1] / rtol
        losses["relative_sum"] = (history / rtol.unsqueeze(-1)).sum(dim=1)

        for k, w in self.weights.items():
            losses["d_loss"] = w * losses[k]

        return losses


class SurrogateSolverLoss(nn.Module):
    def __init__(self, weights: Dict[str, Any]) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, y, y_hat, dvf, dvf_hat):
        losses = {
            "y_loss": mse_loss(y, y_hat),
            "dvf_loss": mse_loss(dvf, dvf_hat),
            "dvf_loss_relative": (((dvf - dvf_hat) / dvf) ** 2).mean(),
            "s_loss": 0,
        }

        for k, w in self.weights.items():
            losses["s_loss"] += w * losses[k]

        return losses


def combine_losses(indepenent_losses, dependent_losses):
    combined_losses = indepenent_losses.copy()
    for k, v in dependent_losses.items():
        if k not in combined_losses:
            combined_losses[k] = v
        else:
            combined_losses[k] += v
    combined_losses["m_loss"] = combined_losses["i_loss"] + combined_losses["d_loss"]

    return combined_losses
