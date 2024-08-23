# %%
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Type

import torch
from petsc4py import PETSc
from tensordict import TensorDict, tensorclass
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
    maxiter: Any = None


@tensorclass(autocast=True)
class PyTorchLinearSystemTask(LinearSystemTask):
    """PyTorch Linear System Task"""

    A: Tensor = None  # currentyl only support dense matrix
    b: Tensor = None
    x: Tensor = None  # Ground Truth if applicable, otherwise the solution provided by the solver
    rtol: Tensor = None
    maxiter: Tensor = None
    is_batched: bool = False


@tensorclass
class PETScLinearSystemTask(LinearSystemTask):
    """PETSc Linear System Task"""

    A: PETSc.Mat | List[PETSc.Mat] = None  # currently only support sparse matrix (AIJ)
    b: PETSc.Vec | List[PETSc.Vec] = None
    x: PETSc.Vec | List[PETSc.Vec] = None  # Ground Truth if applicable, otherwise the solution provided by the solver
    rtol: Tensor = None
    maxiter: Tensor = None
    problem: Any = None  # TODO: Remove this. This is a placeholder for the problem object to keep it alive.
    is_batched: bool = False

    def get_task(self, idx: int) -> LinearSystemTask:
        assert self.is_batched, "This method is only for batched tasks."
        return PETScLinearSystemTask(
            params=self.params[idx],
            A=self.A[idx],
            b=self.b[idx],
            x=self.x[idx],
            rtol=self.rtol[idx],
            maxiter=self.maxiter[idx],
        )


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


## Task related functions ##


def generate_sample_pytorch_task(seed=0) -> PyTorchLinearSystemTask:
    torch.manual_seed(seed)
    params = TensorDict({"param1": torch.randn(5)})
    root_A = torch.randn(5, 5, dtype=torch.float64)
    A = root_A @ root_A.T + 10 * torch.eye(5)  # SPD and diagonally dominant
    x = torch.ones(5, 1, dtype=torch.float64)
    b = A @ x
    rtol = torch.tensor(1.0e-6)
    maxiter = torch.tensor(100)
    return PyTorchLinearSystemTask(params=params, A=A, b=b, x=x, rtol=rtol, maxiter=maxiter)


def generate_sample_petsc_task(seed=0) -> PETScLinearSystemTask:
    pytorch_task = generate_sample_pytorch_task(seed)
    return torch2petsc(pytorch_task)


def generate_sample_batched_pytorch_task(seed=0) -> PyTorchLinearSystemTask:
    pytorch_tasks = [generate_sample_pytorch_task(seed + i) for i in range(3)]
    return pytorch_task_collate_fn(pytorch_tasks)


def generate_sample_batched_petsc_task(seed=0) -> PETScLinearSystemTask:
    petsc_tasks = [generate_sample_petsc_task(seed + i) for i in range(3)]
    return petsc_task_collate_fn(petsc_tasks)


def petsc2torch(task: PETScLinearSystemTask) -> PyTorchLinearSystemTask:
    assert not task.is_batched, "Batched tasks are not supported yet. Please convert them one by one."
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
    assert not task.is_batched, "Batched tasks are not supported yet. Please convert them one by one."
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
    return PETScLinearSystemTask(params=task.params, A=A, b=b, x=x, rtol=task.rtol, maxiter=task.maxiter)


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


def pytorch_task_collate_fn(batch: List[PyTorchLinearSystemTask]) -> PyTorchLinearSystemTask:
    task = torch.stack(batch)
    task.is_batched = True
    return task


def petsc_task_collate_fn(batch: List[PETScLinearSystemTask]) -> PETScLinearSystemTask:
    tau = torch.stack(batch)
    tau.A = [task.A for task in batch]
    tau.b = [task.b for task in batch]
    tau.x = [task.x for task in batch]
    tau.is_batched = True
    return tau


def torch2petsc_collate_fn(batch: List[PyTorchLinearSystemTask]) -> PETScLinearSystemTask:
    batch = [torch2petsc(task) for task in batch]
    return petsc_task_collate_fn(batch)


def petsc2torch_collate_fn(batch: List[PETScLinearSystemTask]) -> PyTorchLinearSystemTask:
    batch = [petsc2torch(task) for task in batch]
    return pytorch_task_collate_fn(batch)
