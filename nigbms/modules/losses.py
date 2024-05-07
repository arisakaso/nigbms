import torch
from torch import Tensor, log, norm, sigmoid
from torch.nn import Module
from torch.nn.functional import mse_loss


class ThetaLoss(Module):
    def __init__(self, weights: dict) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, tau: dict, theta: Tensor):
        rtol = tau["rtol"]  # (bs,)
        bnorm = norm(tau["b"], dim=1)  # (bs,)
        xnorm = norm(tau["x_sol"], dim=1)  # (bs,)
        x0 = theta["x0"]  # (bs, n, 1)
        A = tau["A"]
        b = tau["b"]  # (bs, n, 1)
        x_sol = tau["x_sol"]  # (bs, n, 1)
        r0 = norm(b - A @ x0, dim=(1, 2))  # (bs,)
        e0 = norm(x_sol - x0, dim=(1, 2))  # (bs,)

        losses = {
            "theta_loss": torch.zeros_like(rtol),
            "r0": r0,
            "r0^2": r0**2,
            "relative_r0": r0 / bnorm,
            "relative_r0^2": (r0 / bnorm) ** 2,
            "e0": e0,
            "e0^2": e0**2,
            "relative_e0": e0 / xnorm,
            "relative_e0^2": (e0 / xnorm) ** 2,
        }
        for k, w in self.weights.items():
            losses["i_loss"] = w * losses[k]

        return losses


class HistoryLoss(Module):
    def __init__(self, weights: dict) -> None:
        super().__init__()
        self.weights = weights

    def forward(self, tau: dict, history: Tensor):
        rtol = tau["rtol"]  # (bs,)
        bnorm = norm(tau["b"], dim=1)  # (bs,)
        tol_bnorm = rtol * bnorm

        losses = {
            "history_loss": torch.zeros_like(rtol),
            "iter_r": (history > rtol).sum(dim=1).float(),
            "iter_r_proxy": torch.where(history > tol_bnorm, sigmoid(history - tol_bnorm), 0).sum(dim=1),
            "iter_r_proxy_log": torch.where(history > tol_bnorm, sigmoid(log(history / tol_bnorm)), 0).sum(dim=1),
            "rk": history[:, -1],
            "rk^2": history[:, -1] ** 2,
            "relative_rk": history[:, -1] / bnorm,
        }

        for k, w in self.weights.items():
            losses["history_loss"] = w * losses[k]

        return losses


class SurrogateSolverLoss(Module):
    def __init__(self, weights: dict) -> None:
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


class MetaSolverLoss(Module):
    def __init__(self, weights: dict, reduce: bool) -> None:
        super().__init__()
        self.weights = weights
        self.reduce = reduce

    def forward(self, tau, theta, history):
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
