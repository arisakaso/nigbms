import torch
from omegaconf import DictConfig
from petsc4py import PETSc
from tensordict import TensorDict
from torch import Tensor
from torch.nn import Module

from nigbms.modules.tasks import MinimizeTestFunctionTask, PETScLinearSystemTask, PyTorchLinearSystemTask, Task
from nigbms.utils.convert import tensor2petscvec
from nigbms.utils.solver import clear_petsc_options, eyes_like, set_petsc_options


class _Solver(Module):
    """Base class for solvers."""

    def __init__(self, params_fix: DictConfig, params_learn: DictConfig) -> None:
        """_summary_

        Args:
            params_fix (DictConfig): fixed solver parameters
            params_learn (DictConfig): parameters to be learned
        """
        super().__init__()
        self.params_fix = params_fix
        self.params_learn = params_learn

    def _setup(self, tau: Task, theta: TensorDict) -> None:
        """set up the solver with the given parameters.

        Args:
            tau (Task): Task dataclass
            theta (TensorDict): Dictionary containing solver parameters
        """
        raise NotImplementedError

    def forward(self, tau: Task, theta: TensorDict) -> Tensor:
        """Solve the task with the given parameters.
        Return the convergence history.

        Args:
            tau (Task): Task dataclass
            theta (TensorDict): Dictionary containing solver parameters

        Returns:
            Tensor: convergence history
        """
        raise NotImplementedError


class TestFunctionSolver(_Solver):
    """Solver class for minimizing test functions."""

    def __init__(self, params_fix: DictConfig, params_learn: DictConfig) -> None:
        super().__init__(params_fix, params_learn)

    def forward(self, tau: MinimizeTestFunctionTask, theta: Tensor) -> Tensor:
        return tau.f(theta["x"])


class _PytorchIterativeSolver(_Solver):
    """Base class for iterative solvers written in Pytorch."""

    def __init__(self, params_fix: dict, params_learn: dict) -> None:
        super().__init__(params_fix, params_learn)
        self.history_length = (
            params_fix.history_length
        )  # this is different from maxiter provided in tau, which is used as a stopping criterion.

    def _step(self):
        raise NotImplementedError

    def _setup(self, tau: PyTorchLinearSystemTask, theta: TensorDict):
        """
        Set up the solver with the given parameters.

        Args:
            tau (dict): Dictionary containing task information.
            theta (TensorDict): Dictionary containing solver parameters.

        Returns:
            None
        """

        self.A = tau.A
        self.b = tau.b
        self.x = theta["x0"]
        self.rtol = tau.rtol
        self.maxiter = tau.maxiter
        self.r = self.b - self.A @ self.x
        self.rnorm = torch.norm(self.r, dim=(1, 2))
        self.history = torch.zeros(tau.b.shape[0], self.history_length, device=tau.b.device, dtype=tau.b.dtype)
        self.history[:, 0] = self.rnorm
        self.bnorm = torch.norm(self.b, dim=(1, 2))

    def forward(self, tau: PyTorchLinearSystemTask, theta: TensorDict) -> Tensor:
        """
        Perform forward pass of the solver.

        Args:
            tau (dict): Dictionary containing task information.
            theta (TensorDict): Dictionary containing solver parameters.

        Returns:
            Tensor: The stacked history of absolute residual norm. Shape: (bs, niter, 1)
        """
        self._setup(tau, theta)
        for i in range(self.maxiter.max()):
            self._step()
            self.history[:, i] = self.rnorm

            if all(self.rnorm < self.rtol * self.bnorm):
                break

        if tau.x is None:
            tau.x = self.x  # save the solution if not provided

        return self.history  # (bs, history_length)


class PyTorchJacobi(_PytorchIterativeSolver):
    def __init__(self, params_fix: dict, params_learn: dict) -> None:
        super().__init__(params_fix, params_learn)

    def _setup(self, tau: PyTorchLinearSystemTask, theta: TensorDict):
        super()._setup(tau, theta)
        if "omega" in self.params_fix:
            omega = self.params_fix.omega
        elif "omega" in theta.keys():
            omega = theta.omega
        else:
            omega = 1.0

        D_inv = torch.diag_embed(1 / self.A.diagonal(dim1=1, dim2=2), dim1=1, dim2=2)
        I = eyes_like(self.A)  # noqa: E741
        self.G = I - omega * D_inv @ self.A
        self.f = omega * D_inv @ self.b

    def _step(self):
        self.x = self.G @ self.x + self.f
        self.r = self.b - self.A @ self.x
        self.rnorm = torch.norm(self.r, dim=(1, 2))


class PyTorchSOR(_PytorchIterativeSolver):
    def __init__(self, params_fix: dict, params_learn: dict) -> None:
        super().__init__(params_fix, params_learn)

    def _setup(self, tau: PyTorchLinearSystemTask, theta: TensorDict):
        super()._setup(tau, theta)
        if "omega" in self.params_fix:
            omega = self.params_fix.omega
        elif "omega" in self.params_learn:
            omega = theta["omega"]
        else:
            raise ValueError("omega not found in params_fix or params_learn")

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


class PyTorchCG(_PytorchIterativeSolver):
    def __init__(self, params_fix: dict, params_learn: dict) -> None:
        super().__init__(params_fix, params_learn)

    def _setup(self, tau: PyTorchLinearSystemTask, theta: TensorDict):
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


class PETScKSP(_Solver):
    """Solver class for wrapping PETSc KSP solvers."""

    def __init__(
        self,
        params_fix: dict,
        params_learn: dict,
        debug: bool = False,
    ) -> None:
        super().__init__(params_fix, params_learn)

        # this is different from maxiter provided in tau, which is used as a stopping criterion.
        self.history_length = params_fix.history_length

        # set fixed PETSc options
        self.opts = PETSc.Options()
        clear_petsc_options()
        set_petsc_options(params_fix)

        # set debug options
        if debug:
            PETSc.Log.begin()
            self.opts.setValue("ksp_view", None)
            self.opts.setValue("ksp_converged_reason", None)
            self.opts.setValue("ksp_monitor_true_residual", None)
            self.opts.setValue("log_view", None)
            self.opts.setValue("log_summary", None)
            self.opts.setValue("info", None)

        self.opts.view()

    def solve(self, tau: PETScLinearSystemTask, theta: TensorDict):
        # setup ksp
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setMonitor(lambda ksp, its, rnorm: None)
        self.ksp.setOperators(tau.A)
        self.ksp.setTolerances(rtol=tau.rtol, max_it=tau.maxiter)
        self.ksp.setConvergenceHistory(length=tau.maxiter + 1)

        # set initial guess
        if "x0" in self.params_learn:
            assert self.opts["ksp_initial_guess_nonzero"] == "true"
            x = tensor2petscvec(theta["x0"])
        else:
            x = tau.b.copy()
            x.set(0)

        # solve
        self.ksp.solve(tau.b, x)
        self.x.append(x)
        if tau.x is None:
            tau.x = x  # save the solution if not provided

        # return history
        history = torch.zeros(self.history_length, dtype=theta.dtype, device=theta.device)
        history[: self.ksp.getIterationNumber() + 1] = torch.from_numpy(
            self.ksp.getConvergenceHistory() / tau.b.norm()
        )

        # clean up
        self.ksp.destroy()

        return history

    def forward(self, batched_tau: PyTorchLinearSystemTask, theta: TensorDict) -> Tensor:
        assert batched_tau.is_batched, "batched_tau must be a batched task."
        assert batched_tau.batch_size == theta.batch_size, "batch_size of tau and theta must match."

        self.x = []
        theta = theta.detach().cpu()
        histories = []
        for i in range(len(batched_tau)):
            histories.append(self.solve(batched_tau.get_task(i), theta[i]))
        histories = torch.stack(histories).to(device=theta.device, dtype=theta.dtype)

        return histories


class OpenFOAMSolver(_Solver):  # TODO: implement OpenFOAM solver
    def __init__(
        self,
        params_fix: dict,
        params_learn: dict,
    ) -> None:
        super().__init__(params_fix, params_learn)

        raise NotImplementedError

    def _setup(self, tau: PyTorchLinearSystemTask, theta: TensorDict):
        pass

    def solve(self, A: Tensor, b: Tensor, theta: TensorDict) -> Tensor:
        pass

    def forward(self, tau: PyTorchLinearSystemTask, theta: TensorDict) -> Tensor:
        pass
