import torch
from tensordict import TensorDict
from torch import Tensor

from nigbms.solvers.base import _Solver
from nigbms.utils.solver import eyes_like


class PytorchIterativeSolver(_Solver):
    def __init__(self, params_fix: dict, params_learn: dict) -> None:
        super().__init__(params_fix, params_learn)

    def _step(self):
        raise NotImplementedError

    def _setup(self, tau: dict, theta: TensorDict):
        """
        Set up the solver with the given parameters.

        Args:
            tau (dict): Dictionary containing task information.
            theta (TensorDict): Dictionary containing solver parameters.

        Returns:
            None
        """

        self.A = tau["A"]
        self.b = tau["b"]
        self.x = theta["x0"]
        self.rtol = tau["rtol"]
        self.maxiter = tau["maxiter"]
        self.r = self.b - self.A @ self.x
        self.rnorm = torch.norm(self.r, dim=(1, 2))
        self.history = [self.rnorm]
        self.bnorm = torch.norm(self.b, dim=(1, 2))

    def forward(self, tau: dict, theta: TensorDict) -> Tensor:
        """
        Perform forward pass of the solver.

        Args:
            tau (dict): Dictionary containing task information.
            theta (TensorDict): Dictionary containing solver parameters.

        Returns:
            Tensor: The stacked history of solver outputs. Shape: (bs, niter, 1)
        """
        self._setup(tau, theta)
        for _ in range(self.maxiter.max()):
            self._step()
            self.history.append(self.rnorm)

            if all(self.history[-1] < self.rtol * self.bnorm):
                break

        return torch.stack(self.history, dim=1)  # (bs, niter, 1)


class Jacobi(PytorchIterativeSolver):
    def __init__(self, params_fix: dict, params_learn: dict) -> None:
        super().__init__(params_fix, params_learn)

    def _setup(self, tau: dict, theta: TensorDict):
        super()._setup(tau, theta)
        omega = theta["omega"]

        D_inv = torch.diag_embed(1 / self.A.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
        I = eyes_like(self.A)
        self.G = I - omega * D_inv @ self.A
        self.f = omega * D_inv @ self.b

    def _step(self):
        self.x = self.G @ self.x + self.f
        self.r = self.b - self.A @ self.x
        self.rnorm = torch.norm(self.r, dim=(1, 2))


class SOR(PytorchIterativeSolver):
    def __init__(self, params_fix: dict, params_learn: dict) -> None:
        super().__init__(params_fix, params_learn)

    def _setup(self, tau: dict, theta: TensorDict):
        super()._setup(tau, theta)
        omega = theta["omega"]

        D = torch.diag_embed(self.A.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
        L = torch.tril(self.A, -1)
        U = torch.triu(self.A, 1)
        G1 = D + omega * L
        G2 = omega * U + (omega - 1) * D
        self.G = torch.linalg.solve_triangular(G1, -G2, upper=False)
        self.f = torch.linalg.solve_triangular(G1, self.b * omega, upper=False)

    def _step(self):
        self.x = self.G @ self.x + self.f
        self.r = self.b - self.A @ self.x
        self.rnorm = torch.norm(self.r, dim=(1, 2))


class CG(PytorchIterativeSolver):
    def __init__(self, params_fix: dict, params_learn: dict) -> None:
        super().__init__(params_fix, params_learn)

    def _setup(self, tau: dict, theta: TensorDict):
        super()._setup(tau, theta)
        if "M_inv" in self.params_learn:
            self.M_inv = theta["M_inv"]
        else:
            self.M_inv = eyes_like(self.A)
        self.r_new = self.r.clone()
        self.z = self.M_inv @ self.r
        self.z_new = self.z.clone()
        self.p = self.z.clone()

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
        self.rnorm = torch.norm(self.r, dim=(1, 2))
