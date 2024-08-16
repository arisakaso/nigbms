import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Type

import numpy as np
import torch
from petsc4py import PETSc
from tensordict import TensorDict
from torch import Tensor, sparse_csr_tensor, tensor

from nigbms.utils.distributions import Distribution


@dataclass
class TaskParams:
    """Parameters to generate a task"""

    pass


class TaskDistribution:
    """Base class for task distributions"""

    def __init__(self, task_params_type: Type, distributions: Dict[str, Distribution]):
        self.task_params_type = eval(task_params_type)
        self.distributions = distributions

    def sample(self, seed: int = None) -> TaskParams:
        params = {}
        for key, dist in self.distributions.items():
            params[key] = dist.sample(seed)
        task_params = self.task_params_type(**params)
        return task_params


@dataclass
class Task:
    """Base class for tasks"""

    params: TaskParams = None


@dataclass
class MinimizeTestFunctionTask(Task):
    f: Callable = None  # Test function to minimize


@dataclass
class LinearSystemTask(Task):
    A: Any = None
    b: Any = None
    x: Any = None  # Ground Truth if applicable, otherwise the solution provided by the solver
    rtol: Any = None
    maxiter: Any = None


# TODO: add `is_batched` flag?
@dataclass
class PyTorchLinearSystemTask(LinearSystemTask):
    params: TensorDict = None  # override the base class to use TensorDict
    A: Tensor = None  # currentyl only support dense matrix
    b: Tensor = None
    x: Tensor = None  # Ground Truth if applicable, otherwise the solution provided by the solver
    rtol: Tensor = None
    maxiter: Tensor = None

    def __eq__(self, other) -> bool:  # TODO: add `params` to the comparisonss
        return (
            torch.equal(self.A, other.A)
            and torch.equal(self.b, other.b)
            and ((self.x is None and other.x is None) or torch.equal(self.x, other.x))
            and torch.equal(self.rtol, other.rtol)
            and torch.equal(self.maxiter, other.maxiter)
        )


@dataclass
class PETScLinearSystemTask(LinearSystemTask):
    A: PETSc.Mat = None  # currently only support sparse matrix (AIJ)
    b: PETSc.Vec = None
    x: PETSc.Vec = None  # Ground Truth if applicable, otherwise the solution provided by the solver
    rtol: float = None
    maxiter: int = None
    problem: Any = None  # TODO: Remove this. This is a placeholder for the problem object to keep it alive.

    def __eq__(self, other) -> bool:
        return (
            self.A.equal(other.A)
            and self.b.equal(other.b)
            and ((self.x is None and other.x is None) or self.x.equal(other.x))
            and np.isclose(self.rtol, other.rtol)  # can't use == for float
            and self.maxiter == other.maxiter
        )


@dataclass
class OpenFOAMTask:
    u: Any = None
    p: Any = None


def petsc2torch(task: PETScLinearSystemTask) -> PyTorchLinearSystemTask:
    size = task.A.getSize()
    row_idx, col_idx, values = task.A.getValuesCSR()
    A = sparse_csr_tensor(row_idx, col_idx, values, size).to_dense()

    return PyTorchLinearSystemTask(
        params=task.params,
        A=A,
        b=tensor(task.b.getArray()),
        x=tensor(task.x.getArray()) if task.x is not None else None,
        rtol=tensor(task.rtol),
        maxiter=tensor(task.maxiter),
    )


def torch2petsc(task: PyTorchLinearSystemTask) -> PETScLinearSystemTask:
    A_sp = task.A.cpu().to_sparse_csr()
    A = PETSc.Mat().createAIJ(size=A_sp.shape, nnz=A_sp._nnz())
    A.setValuesCSR(
        I=A_sp.crow_indices().numpy().astype("int32"),
        J=A_sp.col_indices().numpy().astype("int32"),
        V=A_sp.values().numpy(),
    )
    A.assemble()
    b = PETSc.Vec().createWithArray(task.b.numpy())
    x = PETSc.Vec().createWithArray(task.x.numpy()) if task.x is not None else None
    return PETScLinearSystemTask(params=task.params, A=A, b=b, x=x, rtol=float(task.rtol), maxiter=int(task.maxiter))


def generate_sample_pytorch_task(seed=0) -> PyTorchLinearSystemTask:
    torch.manual_seed(seed)
    params = torch.randn(5)
    root_A = torch.randn(5, 5, dtype=torch.float64)
    A = root_A @ root_A.T + 10 * torch.eye(5)  # SPD and diagonally dominant
    x = torch.ones(5, 1, dtype=torch.float64)
    b = A @ x
    rtol = torch.tensor(1.0e-6)
    maxiter = torch.tensor(100)
    return PyTorchLinearSystemTask(params=params, A=A, b=b, x=x, rtol=rtol, maxiter=maxiter)


def save_petsc_task(task: PETScLinearSystemTask, path: Path) -> None:
    """Save PETScLinearSystemTask to disk.
    `pickle.dump` cannot be used to save PETSc objects.

    Args:
        task (PETScLinearSystemTask): task
        idx (int): index
    """
    path.mkdir(parents=True, exist_ok=True)
    viewer_A = PETSc.Viewer().createBinary(str(path / "A.dat"), "w")
    task.A.view(viewer_A)
    viewer_A.destroy()

    viewer_b = PETSc.Viewer().createBinary(str(path / "b.dat"), "w")
    task.b.view(viewer_b)
    viewer_b.destroy()

    if task.x is not None:
        viewer_x = PETSc.Viewer().createBinary(str(path / "x.dat"), "w")
        task.x.view(viewer_x)
        viewer_x.destroy()

    pickle.dump(task.params, (path / "params.pkl").open("wb"))


def load_petsc_task(path: Path) -> PETScLinearSystemTask:
    """Load PETScLinearSystemTask from disk.
    Note that rtol and maxiter are not saved, and they need to be set later.

    Args:
        path (Path): path to the task

    Returns:
        PETScLinearSystemTask: task
    """
    viewer_A = PETSc.Viewer().createBinary(str(path / "A.dat"), "r")
    A = PETSc.Mat().create()
    A.load(viewer_A)
    viewer_A.destroy()

    viewer_b = PETSc.Viewer().createBinary(str(path / "b.dat"), "r")
    b = PETSc.Vec().create()
    b.load(viewer_b)
    viewer_b.destroy()

    if (path / "x.dat").exists():
        viewer_x = PETSc.Viewer().createBinary(str(path / "x.dat"), "r")
        x = PETSc.Vec().create()
        x.load(viewer_x)
        viewer_x.destroy()
    else:
        x = None

    params = pickle.load((path / "params.pkl").open("rb"))

    return PETScLinearSystemTask(params=params, A=A, b=b, x=x, rtol=None, maxiter=None)


def save_pytorch_task(path: Path) -> None:
    pass


def load_pytorch_task(path: Path) -> PyTorchLinearSystemTask:
    pass
