# %%
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Type

import torch
from petsc4py import PETSc
from tensordict import tensorclass
from torch import Tensor, sparse_csr_tensor, tensor

from nigbms.utils.distributions import Distribution


@tensorclass
class TaskParams:
    """Parameters to generate a task"""

    pass


@tensorclass
class Task:
    """Base class for tasks"""

    params: TaskParams | None = None


@tensorclass
class MinimizeTestFunctionTask(Task):
    f: Callable = None  # Test function to minimize


@tensorclass
class LinearSystemTask(Task):
    A: Any = None
    b: Any = None
    x: Any = None  # Ground Truth if applicable, otherwise the solution provided by the solver
    rtol: Any = None
    maxiter: Any = None


# TODO: add `is_batched` flag?
@tensorclass
class PyTorchLinearSystemTask(LinearSystemTask):
    """PyTorch Linear System Task"""

    A: Tensor = None  # currentyl only support dense matrix
    b: Tensor = None
    x: Tensor = None  # Ground Truth if applicable, otherwise the solution provided by the solver
    rtol: Tensor = None
    maxiter: Tensor = None


@tensorclass
class PETScLinearSystemTask(LinearSystemTask):
    """PETSc Linear System Task"""

    A: PETSc.Mat = None  # currently only support sparse matrix (AIJ)
    b: PETSc.Vec = None
    x: PETSc.Vec = None  # Ground Truth if applicable, otherwise the solution provided by the solver
    rtol: float = None
    maxiter: int = None
    problem: Any = None  # TODO: Remove this. This is a placeholder for the problem object to keep it alive.


@tensorclass
class OpenFOAMTask:
    u: Any = None
    p: Any = None


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


def save_pytorch_task(task: PyTorchLinearSystemTask, path: Path) -> None:
    """Save PyTorchLinearSystemTask to disk."""
    # task.memmap(path)
    path.mkdir(parents=True, exist_ok=True)
    pickle.dump(task, (path / "task.pkl").open("wb"))


def load_pytorch_task(path: Path) -> PyTorchLinearSystemTask:
    """Load PyTorchLinearSystemTask from disk."""
    # FIXME: load_memmap gives an error: /usr/local/lib/python3.10/dist-packages/tensordict/_td.py:2390: KeyError
    # return TensorDict.load_memmap(path)
    return pickle.load((path / "task.pkl").open("rb"))


# %%
