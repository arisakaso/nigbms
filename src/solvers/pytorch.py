# %%
from typing import Dict

import torch

from src.utils.utils import extract_param


class Solver(torch.nn.Module):
    def __init__(self, params_fix: Dict, params_learn: Dict) -> None:
        super().__init__()
        self.params_fix = params_fix

        # corresponding indices of theta
        idx = 0
        for v in params_learn.values():
            v.start = idx
            v.end = idx + v.dim
            idx += v.dim
        self.params_learn = params_learn

    def _setup(self, tau: Dict, theta: torch.Tensor):
        raise NotImplementedError

    def forward(self, tau: Dict, theta: torch.Tensor):
        raise NotImplementedError


class PytorchSolver(Solver):
    def __init__(self, params_fix: Dict, params_learn: Dict, early_stop=False) -> None:
        super().__init__(params_fix, params_learn)
        self.early_stop = early_stop

    def _step(self):
        raise NotImplementedError

    def _setup(self, tau: Dict, theta: torch.Tensor):
        """task dependent solver setup"""
        self.maxiter = tau["maxiter"].max()
        self.rtol = tau["rtol"]

        bs, n, _ = tau["A"].shape
        self.A = tau["A"]  # (bs, n, n)
        self.b = tau["b"].reshape(bs, n, 1)  # (bs, n, 1)
        if "x0" in self.params_learn:
            self.x = extract_param("x0", self.params_learn, theta).reshape(
                -1, n, 1
            )  # (bs, n, 1)
            assert self.x.shape == self.b.shape
        else:
            self.x = torch.zeros_like(self.b)
        self.history = []

        if "x_sol" in tau:  # if solution is availabe
            self.x_sol = tau["x_sol"].reshape(bs, n, 1)  # (bs, n, 1)
            self.error = torch.norm(self.x_sol - self.x, dim=(1, 2))
            self.xtol = torch.norm(self.x_sol, dim=(1, 2)) * self.rtol  # (bs,)
            self.residual = self.btol = None
        else:
            self.residual = torch.norm(self.b - self.A @ self.x, dim=(1, 2))
            self.btol = torch.norm(self.b, dim=(1, 2)) * self.rtol  # (bs,)
            self.x_sol = self.error = self.xtol = None

    def forward(self, tau: Dict, theta: torch.Tensor):
        """return entire solution sequence.
        This is not efficient. Just for study purpose.

        Args:
            tau (Dict[str, torch.Tensor]): _description_
            theta (Dict[str, torch.Tensor]): _description_

        Returns:
            _type_: _description_
        """
        self._setup(tau, theta)
        for i in range(self.maxiter):
            self._step()
            self.history.append(self.x)

            # stop before maxiter if all residual is smaller than btol
            if self.early_stop:
                if self.x_sol is not None:
                    if all(self.error < self.xtol):
                        return torch.stack(self.history, dim=1)
                else:
                    if all(self.residual < self.btol):
                        return torch.stack(self.history, dim=1)

        return torch.stack(self.history, dim=1)  # (bs, niter, n, 1)


class Jacobi(PytorchSolver):
    def __init__(self, params_fix: Dict, params_learn: Dict, early_stop=False) -> None:
        super().__init__(params_fix, params_learn, early_stop=early_stop)

    def _setup(self, tau: Dict, theta: torch.Tensor):
        super()._setup(tau, theta)

        D_inv = torch.diag_embed(1 / self.A.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
        L = torch.tril(self.A, -1)
        U = torch.triu(self.A, 1)
        self.G = -D_inv @ (L + U)
        self.f = D_inv @ self.b

    def _step(self):
        self.x = self.G @ self.x + self.f
        self.residual = torch.norm(self.b - self.A @ self.x, dim=(1, 2))
        if self.x_sol is not None:
            self.error = torch.norm(self.x_sol - self.x, dim=(1, 2))


class SOR(Solver):
    def __init__(self, params_fix: Dict, theta_idx: Dict) -> None:
        super().__init__(params_fix, theta_idx)
        if "omega" in params_fix:
            self.omega = params_fix["omega"]
        else:
            self.omega = None

    def _setup(self, tau, theta):
        super()._setup(tau, theta)
        bs, n, _ = self.b.shape

        D = torch.diag_embed(self.A.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
        D_inv = torch.diag_embed(1 / self.A.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
        L = torch.tril(self.A, -1)
        U = torch.triu(self.A, 1)
        I = (
            torch.eye(n, n, dtype=self.b.dtype, device=self.b.device)
            .unsqueeze(0)
            .repeat(bs, 1, 1)
        )
        if self.omega is not None:
            omega = self.omega
        else:
            omega = theta["omega"].reshape(-1, 1, 1)

        G1 = I + omega * D_inv @ L
        G2 = (1 - omega) * I - omega * D_inv @ U
        self.G = torch.linalg.solve_triangular(G1, G2, upper=False)
        self.f = (
            torch.linalg.solve_triangular(D + omega * L, self.b, upper=False) * omega
        )

    def _step(self):
        self.x = self.G @ self.x + self.f
        self.residual = torch.norm(self.b - self.A @ self.x, dim=(1, 2))
        if self.x_sol is not None:
            self.error = torch.norm(self.x_sol - self.x, dim=(1, 2))


class CG(PytorchSolver):
    def __init__(self, params_fix: Dict, params_learn: Dict, early_stop=False) -> None:
        super().__init__(params_fix, params_learn, early_stop=early_stop)

    def _setup(self, tau: Dict, theta: torch.Tensor):
        super()._setup(tau, theta)
        bs, n, _ = self.b.shape
        if "M_inv" in self.params_learn:
            self.M_inv = self._extract_param(theta, "M_inv").reshape(bs, n, n)
        else:
            self.M_inv = (
                torch.eye(n, n, dtype=self.b.dtype, device=self.b.device)
                .unsqueeze(0)
                .repeat(bs, 1, 1)
            )
            # self.M_inv = torch.diag_embed(1 / self.A.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
        self.r = self.b - self.A @ self.x
        self.r_new = self.r.clone()
        self.z = self.M_inv @ self.r
        self.z_new = self.z.clone()
        self.p = self.z.clone()
        self.residual = torch.norm(self.r, dim=(1, 2))

    def _step(self):
        Ap = self.A @ self.p
        alpha = (self.r.mT @ self.z) / (self.p.mT @ Ap)
        self.x = self.x + alpha * self.p
        self.r_new = self.r - alpha * Ap
        self.z_new = self.M_inv @ self.r_new
        beta = (self.r_new.mT @ self.z_new) / (self.r.mT @ self.z)
        self.p = self.z_new + beta * self.p

        self.r = self.r_new
        self.z = self.z_new

        if self.x_sol is not None:
            self.error = torch.norm(self.x_sol - self.x, dim=(1, 2))
        else:
            self.residual = torch.norm(self.r, dim=(1, 2))


# %%
