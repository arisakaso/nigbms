from dataclasses import dataclass
from typing import Any, Callable

from petsc4py import PETSc
from torch import Tensor


@dataclass
class TaskParams:
    """Parameters to generate a task"""

    pass


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


@dataclass
class PyTorchLinearSystemTask(LinearSystemTask):
    A: Tensor = None
    b: Tensor = None
    x: Tensor = None  # Ground Truth if applicable, otherwise the solution provided by the solver
    rtol: Tensor = None
    maxiter: Tensor = None


@dataclass
class PETScLinearSystemTask(LinearSystemTask):
    A: PETSc.Mat = None
    b: PETSc.Vec = None
    x: PETSc.Vec = None  # Ground Truth if applicable, otherwise the solution provided by the solver
    rtol: float = None
    maxiter: int = None


@dataclass
class OpenFOAMTask:
    u: Any = None
    p: Any = None
