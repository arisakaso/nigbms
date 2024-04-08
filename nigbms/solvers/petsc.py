import torch
from joblib import Parallel, delayed
from petsc4py import PETSc
from tensordict import TensorDict
from torch import Tensor

from nigbms.solvers.base import _Solver
from nigbms.utils.convert import tensor2petscvec, torchcoo2petscmat
from nigbms.utils.solver import clear_petsc_options, set_petsc_options


class PETScKSPSolver(_Solver):
    def __init__(
        self,
        params_fix: dict,
        params_learn: dict,
        debug: bool = False,
        parallel: bool = False,
    ) -> None:
        super().__init__(params_fix, params_learn)
        self.parallel = parallel
        self.opts = PETSc.Options()

        # set fixed PETSc options
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

    def _setup(self, tau: dict, theta: TensorDict):
        pass

    def solve(self, A: Tensor, b: Tensor, theta: TensorDict, rtol, maxiter):
        A = torchcoo2petscmat(A)
        b = tensor2petscvec(b)

        # set initial guess
        if "x0" in self.params_learn:
            assert self.opts["ksp_initial_guess_nonzero"] == "true"
            x0 = tensor2petscvec(theta["x0"])
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
        history[: self.ksp.getIterationNumber() + 1] = torch.from_numpy(self.ksp.getConvergenceHistory() / b.norm())

        return history

    def forward(self, tau: dict, theta: TensorDict) -> Tensor:
        A = tau["A"].cpu()
        b = tau["b"].cpu()
        rtol = tau["rtol"].cpu().numpy()
        maxiter = tau["maxiter"].cpu().numpy()
        theta = theta.detach().cpu()
        if A.layout != torch.sparse_coo:
            A = A.to_sparse_coo()

        # setup ksp
        self.ksp = PETSc.KSP().create()
        self.ksp.setFromOptions()
        self.ksp.setMonitor(lambda ksp, its, rnorm: None)

        if self.parallel:
            raise NotImplementedError
            outputs = Parallel(n_jobs=-1)(
                delayed(self.solve)(A[i], b[i], theta[i], rtol[i], maxiter[i]) for i in range(len(b))
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
