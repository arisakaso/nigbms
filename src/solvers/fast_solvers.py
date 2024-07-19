# %%
from typing import Any, Dict

import torch
from torch import Tensor


class JacobiJit(torch.nn.Module):
    def __init__(
        self, params_fix: Dict[str, Any], params_learn: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.params_fix = params_fix

        # corresponding indices of theta
        idx = 0
        for v in params_learn.values():
            v.start = idx
            v.end = idx + v.dim
            idx += v.dim
        self.params_learn = params_learn
        self.criterion = params_fix["criterion"]

        if "omega" in params_fix:
            self.omega = params_fix["omega"]
        else:
            self.omega = None

        self.jittable = True
        self.maxiter = 0
        self.rtol = torch.tensor(0)
        self.A = torch.tensor(0)
        self.b = torch.tensor(0)
        self.x = torch.tensor(0)
        self.r = torch.tensor(0)
        self.G = torch.tensor(0)
        self.L = torch.tensor(0)
        self.U = torch.tensor(0)
        self.f = torch.tensor(0)
        self.x_sol = torch.tensor(0)
        self.relative_error = torch.tensor(0)
        self.relative_residual = torch.tensor(0)
        self.xnorm = torch.tensor(0)
        self.bnorm = torch.tensor(0)
        self.history = torch.tensor(0)

    def forward(self, tau: Dict[str, Tensor], theta: Tensor):
        """return entire solution sequence.
        This is not efficient. Just for study purpose.

        Args:
            tau (Dict[str, Tensor]): _description_
            theta (Dict[str, Tensor]): _description_

        Returns:
            _type_: _description_
        """
        # SETUP
        _device = tau["A"].device
        self.maxiter = int(tau["maxiter"].max())
        self.rtol = tau["rtol"]

        bs, n, _ = tau["A"].shape
        self.A = tau["A"][:1, :, :]  # (1, n, n)
        self.b = tau["b"].reshape(bs, n, 1)  # (bs, n, 1)
        self.x = theta.reshape(bs, n, 1)
        self.r = self.b - self.A @ self.x

        self.x_sol = tau["x_sol"].reshape(bs, n, 1)  # (bs, n, 1)
        self.xnorm = torch.norm(self.x_sol, dim=(1, 2))
        self.bnorm = torch.norm(self.b, dim=(1, 2))
        self.relative_error = torch.norm(self.x_sol - self.x, dim=(1, 2)) / self.xnorm
        self.relative_residual = torch.norm(self.r, dim=(1, 2)) / self.bnorm
        self.history = torch.zeros(
            bs, self.maxiter + 1, dtype=self.x.dtype, device=_device
        )

        D_inv = torch.diag_embed(1 / self.A.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
        I = torch.eye(
            self.A.shape[1], dtype=self.A.dtype, device=self.A.device
        ).unsqueeze(0)
        self.G = I - self.omega * D_inv @ self.A
        self.f = self.omega * D_inv @ self.b

        if self.criterion == "relative_error":
            criterion = self.relative_error
        else:
            criterion = self.relative_residual

        # COMPUTE
        i = 0

        self.history[:, i] = criterion
        while i < self.maxiter and any(criterion > self.rtol):
            self.x = self.G @ self.x + self.f
            self.r = self.b - self.A @ self.x

            self.relative_error = (
                torch.norm(self.x_sol - self.x, dim=(1, 2)) / self.xnorm
            )
            self.relative_residual = torch.norm(self.r, dim=(1, 2)) / self.bnorm

            if self.criterion == "relative_error":
                criterion = self.relative_error
            else:
                criterion = self.relative_residual
            i += 1
            self.history[:, i] = criterion

        return self.history


class SORJit(torch.nn.Module):
    def __init__(
        self, params_fix: Dict[str, Any], params_learn: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.params_fix = params_fix
        if "omega" in params_fix:
            self.omega = params_fix["omega"]
        else:
            self.omega = None

        # corresponding indices of theta
        idx = 0
        for v in params_learn.values():
            v.start = idx
            v.end = idx + v.dim
            idx += v.dim
        self.params_learn = params_learn

        self.maxiter = 0
        self.rtol = torch.tensor(0)
        self.A = torch.tensor(0)
        self.b = torch.tensor(0)
        self.x = torch.tensor(0)
        self.G = torch.tensor(0)
        self.L = torch.tensor(0)
        self.U = torch.tensor(0)
        self.f = torch.tensor(0)
        self.x_sol = torch.tensor(0)
        self.error = torch.tensor(0)
        self.xtol = torch.tensor(0)
        self.history_err = torch.tensor(0)

    def forward(self, tau: Dict[str, Tensor], theta: Tensor):
        """return entire solution sequence.
        This is not efficient. Just for study purpose.

        Args:
            tau (Dict[str, Tensor]): _description_
            theta (Dict[str, Tensor]): _description_

        Returns:
            _type_: _description_
        """
        # SETUP
        self.maxiter = int(tau["maxiter"].max())
        self.rtol = tau["rtol"]

        bs, n, _ = tau["A"].shape
        self.A = tau["A"]  # (bs, n, n)
        self.b = tau["b"].reshape(bs, n, 1)  # (bs, n, 1)
        self.x = theta.reshape(bs, n, 1)

        self.x_sol = tau["x_sol"].reshape(bs, n, 1)  # (bs, n, 1)
        self.error = torch.norm(self.x_sol - self.x, dim=(1, 2))
        self.xtol = torch.norm(self.x_sol, dim=(1, 2)) * self.rtol  # (bs,)
        self.history_err = torch.zeros(
            bs, self.maxiter + 1, dtype=self.x.dtype, device=self.x.device
        )

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
        G1 = D + omega * L
        G2 = omega * U + (omega - 1) * D
        self.G = torch.linalg.solve_triangular(G1, -G2, upper=False)
        self.f = torch.linalg.solve_triangular(G1, self.b * omega, upper=False)

        # COMPUTE
        i = 0
        self.history_err[:, i] = self.error
        while i < self.maxiter and any(self.error > self.xtol):
            self.x = self.G @ self.x + self.f
            self.error = torch.norm(self.x_sol - self.x, dim=(1, 2))
            i += 1
            self.history_err[:, i] = self.error

        return self.history_err


@torch.jit.script
def jacobi(A: Tensor, b: Tensor, x: Tensor, omega: float, maxiter: int):
    D_inv = torch.diag_embed(1 / A.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
    I = torch.eye(A.shape[1], dtype=A.dtype, device=A.device).unsqueeze(0)
    G = I - omega * D_inv @ A
    f = omega * D_inv @ b
    for _ in range(maxiter):
        x = G @ x + f
    return x


class MultigridJit(torch.nn.Module):
    def __init__(
        self, params_fix: Dict[str, Any], params_learn: Dict[str, Any]
    ) -> None:
        super().__init__()
        self.params_fix = params_fix

        # corresponding indices of theta
        idx = 0
        for v in params_learn.values():
            v.start = idx
            v.end = idx + v.dim
            idx += v.dim
        self.params_learn = params_learn

        self.jittable = True
        self.criterion = params_fix["criterion"]
        self.omega = params_fix["omega"]
        self.niters = list(params_fix["niters"])
        self.maxiter = 0
        self.rtol = torch.tensor(0)
        self.A = torch.tensor(0)
        self.b = torch.tensor(0)
        self.x = torch.tensor(0)
        self.r = torch.tensor(0)
        self.R = torch.tensor(0)
        self.P = torch.tensor(0)
        self.A_c = torch.tensor(0)
        self.r_c = torch.tensor(0)
        self.e_c = torch.tensor(0)
        self.e = torch.tensor(0)
        self.x_sol = torch.tensor(0)
        self.relative_error = torch.tensor(0)
        self.relative_residual = torch.tensor(0)
        self.xnorm = torch.tensor(0)
        self.bnorm = torch.tensor(0)
        self.history = torch.tensor(0)

    def forward(self, tau: Dict[str, Tensor], theta: Tensor):
        """return entire solution sequence.
        This is not efficient. Just for study purpose.

        Args:
            tau (Dict[str, Tensor]): _description_
            theta (Dict[str, Tensor]): _description_

        Returns:
            _type_: _description_
        """
        # SETUP
        _device = tau["A"].device
        self.maxiter = int(tau["maxiter"].max())
        self.rtol = tau["rtol"]

        bs, n, _ = tau["A"].shape
        self.A = tau["A"][:1, :, :]  # (1, n, n)
        self.b = tau["b"].reshape(bs, n, 1)  # (bs, n, 1)
        self.x = theta.reshape(bs, n, 1)
        self.r = self.b - self.A @ self.x

        self.x_sol = tau["x_sol"].reshape(bs, n, 1)  # (bs, n, 1)
        self.xnorm = torch.norm(self.x_sol, dim=(1, 2))
        self.bnorm = torch.norm(self.b, dim=(1, 2))
        self.relative_error = torch.norm(self.x_sol - self.x, dim=(1, 2)) / self.xnorm
        self.relative_residual = torch.norm(self.r, dim=(1, 2)) / self.bnorm
        self.history = torch.zeros(
            bs, self.maxiter + 1, dtype=self.x.dtype, device=_device
        )

        if self.criterion == "relative_error":
            criterion = self.relative_error
        else:
            criterion = self.relative_residual

        self.R, self.P = make_R_and_P(self.A)
        self.A_c = self.R @ self.A @ self.P

        # COMPUTE
        i = 0
        self.history[:, i] = criterion
        while i < self.maxiter and any(criterion > self.rtol):
            # pre-sommoothing on fine grid
            self.x = jacobi(self.A, self.b, self.x, self.omega, self.niters[0])
            self.r = self.b - self.A @ self.x

            # coarse grid
            self.r_c = self.R @ self.r  # restriction
            self.e_c = jacobi(
                self.A_c,
                self.r_c,
                torch.zeros_like(self.r_c),
                self.omega,
                self.niters[1],
            )

            # correction
            self.e = self.P @ self.e_c  # prolongation
            self.x = self.x + self.e  # correction

            # post-smoothing on fine grid
            self.x = jacobi(self.A, self.b, self.x, self.omega, self.niters[0])
            self.r = self.b - self.A @ self.x

            self.relative_error = (
                torch.norm(self.x_sol - self.x, dim=(1, 2)) / self.xnorm
            )
            self.relative_residual = torch.norm(self.r, dim=(1, 2)) / self.bnorm
            if self.criterion == "relative_error":
                criterion = self.relative_error
            else:
                criterion = self.relative_residual

            i += 1
            self.history[:, i] = criterion

        return self.history


@torch.jit.script
def make_R_and_P(A: Tensor):
    R = torch.abs(A[:, 1::2]).reshape(1, A.shape[1] // 2, A.shape[1]) / 4
    P = 2 * R.transpose(1, 2)
    return R, P
