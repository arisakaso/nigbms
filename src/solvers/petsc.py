# %%
from typing import Any, Dict

import torch
from joblib import Parallel, delayed
from petsc4py import PETSc
from torch import Tensor

from src.solvers.fast_solvers import make_R_and_P
from src.solvers.pytorch import Solver
from src.utils.utils import tensor2petscvec, torchcoo2petscmat


def clear_petsc_options():
    opts = PETSc.Options()
    for key in opts.getAll().keys():
        opts.delValue(key)


def set_petsc_options(opts: Dict[str, Any]):
    for key, value in opts.items():
        PETSc.Options().setValue(key, value)
    PETSc.Options().view()


class PETScGMGSolver(Solver):
    def __init__(
        self,
        params_fix: Dict[str, Any],
        params_learn: Dict[str, Any],
        debug: bool = False,
        parallel: bool = False,
    ):
        super().__init__(params_fix, params_learn)
        self.parallel = parallel
        self.jittable = False

        # set common PETSc options
        clear_petsc_options()
        opts = PETSc.Options()
        for key, value in params_fix.items():
            opts.setValue(key, value)
        if debug:  # set debug options
            PETSc.Log.begin()
            opts.setValue("ksp_view", None)
            opts.setValue("ksp_converged_reason", None)
            opts.setValue("ksp_monitor_true_residual", None)
            opts.setValue("log_view", None)
            opts.setValue("log_summary", None)
            opts.setValue("info", None)
        opts.view()

    def solve(self, b: Tensor, theta: Tensor, rtol, maxiter):
        b = tensor2petscvec(b)

        # set initial guess
        if "x0" in self.params_learn:
            # assert opts["ksp_initial_guess_nonzero"] == "true"
            # x0 = extract_param("x0", self.params_learn, theta)
            x0 = tensor2petscvec(theta)
        else:
            x0 = b.copy()
            x0.set(0)

        # setup ksp
        self.ksp.setTolerances(rtol=rtol, max_it=maxiter)
        self.ksp.setConvergenceHistory()

        # solve
        self.ksp.solve(b, x0)

        # return history
        history = torch.zeros(maxiter + 1, dtype=theta.dtype, device=theta.device)
        history[: self.ksp.getIterationNumber() + 1] = torch.from_numpy(
            self.ksp.getConvergenceHistory() / b.norm()
        )

        return history

    def forward(self, tau: Dict[str, Any], theta: Tensor):
        A = tau["A"][0].cpu()
        if A.layout != torch.sparse_coo:
            A = A.to_sparse_coo()
        b = tau["b"].cpu()
        rtol = tau["rtol"].cpu().numpy()
        maxiter = tau["maxiter"].cpu().numpy()
        theta = theta.detach().cpu()

        A = torchcoo2petscmat(A)
        R, P = make_R_and_P(tau["A"][:1])
        R = R.squeeze()
        P = P.squeeze()
        R = torchcoo2petscmat(R.to_sparse_coo())
        P = torchcoo2petscmat(P.to_sparse_coo())
        A1 = R @ A @ P

        # setup ksp
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setOperators(A)
        self.ksp.setMonitor(lambda ksp, its, rnorm: None)
        pc = self.ksp.getPC()
        pc.setMGRestriction(1, R)
        pc.setMGInterpolation(1, P)
        ksp0 = pc.getMGCoarseSolve()
        ksp0.setOperators(A1)

        if self.parallel:
            raise NotImplementedError
            outputs = Parallel(n_jobs=-1)(
                delayed(self.solve)(A[i], b[i], theta[i], rtol[i], maxiter[i])
                for i in range(len(b))
            )
            history = torch.stack(outputs)

        else:
            history = []
            for i in range(len(b)):
                history.append(self.solve(b[i], theta[i], rtol[i], maxiter[i]))
            history = torch.stack(history)

        history = history.to(device=tau["b"].device, dtype=tau["b"].dtype)

        self.ksp.destroy()
        return history


class PETScSolverFixedA(Solver):
    def __init__(
        self,
        params_fix: Dict[str, Any],
        params_learn: Dict[str, Any],
        debug: bool = False,
        parallel: bool = False,
    ):
        super().__init__(params_fix, params_learn)
        self.parallel = parallel
        self.jittable = False

        # set common PETSc options
        clear_petsc_options()
        opts = PETSc.Options()
        for key, value in params_fix.items():
            opts.setValue(key, value)
        if debug:  # set debug options
            PETSc.Log.begin()
            opts.setValue("ksp_view", None)
            opts.setValue("ksp_converged_reason", None)
            opts.setValue("ksp_monitor_true_residual", None)
            opts.setValue("log_view", None)
            opts.setValue("log_summary", None)
            opts.setValue("info", None)
        opts.view()

    def solve(self, b: Tensor, theta: Tensor, rtol, maxiter):
        b = tensor2petscvec(b)

        # set initial guess
        if "x0" in self.params_learn:
            # assert opts["ksp_initial_guess_nonzero"] == "true"
            # x0 = extract_param("x0", self.params_learn, theta)
            x0 = tensor2petscvec(theta)
        else:
            x0 = b.copy()
            x0.set(0)

        # setup ksp
        self.ksp.setTolerances(rtol=rtol, max_it=maxiter)
        self.ksp.setConvergenceHistory(length=maxiter + 1)

        # solve
        self.ksp.solve(b, x0)

        # return history
        history = torch.zeros(maxiter + 1, dtype=theta.dtype, device=theta.device)
        history[: self.ksp.getIterationNumber() + 1] = torch.from_numpy(
            self.ksp.getConvergenceHistory() / b.norm()
        )

        return history

    def forward(self, tau: Dict[str, Any], theta: Tensor):
        A = tau["A"][0].cpu()
        if A.layout != torch.sparse_coo:
            A = A.to_sparse_coo()
        b = tau["b"].cpu()
        rtol = tau["rtol"].cpu().numpy()
        maxiter = tau["maxiter"].cpu().numpy()
        theta = theta.detach().cpu()

        A = torchcoo2petscmat(A)

        # setup ksp
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setOperators(A)  # fix A
        self.ksp.setMonitor(lambda ksp, its, rnorm: None)

        if self.parallel:
            raise NotImplementedError
            outputs = Parallel(n_jobs=-1)(
                delayed(self.solve)(A[i], b[i], theta[i], rtol[i], maxiter[i])
                for i in range(len(b))
            )
            history = torch.stack(outputs)

        else:
            history = []
            for i in range(len(b)):
                history.append(self.solve(b[i], theta[i], rtol[i], maxiter[i]))
            history = torch.stack(history)

        history = history.to(device=tau["b"].device, dtype=tau["b"].dtype)

        self.ksp.destroy()
        return history


# %%
class PETScDefaultSolver(Solver):
    def __init__(
        self,
        params_fix: Dict[str, Any],
        params_learn: Dict[str, Any],
        debug: bool = False,
        parallel: bool = False,
    ):
        super().__init__(params_fix, params_learn)
        self.parallel = parallel
        self.jittable = False

        # set common PETSc options
        clear_petsc_options()
        opts = PETSc.Options()
        for key, value in params_fix.items():
            opts.setValue(key, value)
        if debug:  # set debug options
            PETSc.Log.begin()
            opts.setValue("ksp_view", None)
            opts.setValue("ksp_converged_reason", None)
            opts.setValue("ksp_monitor_true_residual", None)
            opts.setValue("log_view", None)
            opts.setValue("log_summary", None)
            opts.setValue("info", None)
        opts.view()

    def solve(self, A: Tensor, b: Tensor, theta: Tensor, rtol, maxiter):
        A = torchcoo2petscmat(A)
        b = tensor2petscvec(b)

        # set initial guess
        if "x0" in self.params_learn:
            # assert opts["ksp_initial_guess_nonzero"] == "true"
            # x0 = extract_param("x0", self.params_learn, theta)
            x0 = tensor2petscvec(theta)
        else:
            x0 = b.copy()
            x0.set(0)

        # setup ksp
        self.ksp.setOperators(A)
        self.ksp.setTolerances(rtol=rtol, max_it=maxiter)
        self.ksp.setConvergenceHistory(length=maxiter + 1)

        # solve
        self.ksp.solve(b, x0)

        # return history
        history = torch.zeros(maxiter + 1, dtype=theta.dtype, device=theta.device)
        history[: self.ksp.getIterationNumber() + 1] = torch.from_numpy(
            self.ksp.getConvergenceHistory() / b.norm()
        )

        return history

    def forward(self, tau: Dict[str, Any], theta: Tensor):
        A = tau["A"].cpu()
        if A.layout != torch.sparse_coo:
            A = A.to_sparse_coo()
        b = tau["b"].cpu()
        rtol = tau["rtol"].cpu().numpy()
        maxiter = tau["maxiter"].cpu().numpy()
        theta = theta.detach().cpu()

        # setup ksp
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setMonitor(lambda ksp, its, rnorm: None)

        if self.parallel:
            raise NotImplementedError
            outputs = Parallel(n_jobs=-1)(
                delayed(self.solve)(A[i], b[i], theta[i], rtol[i], maxiter[i])
                for i in range(len(b))
            )
            history = torch.stack(outputs)

        else:
            history = []
            for i in range(len(b)):
                history.append(self.solve(A[i], b[i], theta[i], rtol[i], maxiter[i]))
            history = torch.stack(history)

        history = history.to(device=tau["b"].device, dtype=tau["b"].dtype)

        self.ksp.destroy()
        return history
