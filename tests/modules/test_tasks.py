import pytest
import torch
from petsc4py import PETSc

from nigbms.modules.tasks import (
    PETScLinearSystemTask,
    PyTorchLinearSystemTask,
    generate_sample_pytorch_task,
    petsc2torch,
    torch2petsc,
)


@pytest.fixture
def petsc_task() -> PETScLinearSystemTask:
    A = PETSc.Mat().createAIJ([3, 3])
    A.setUp()
    A[0, 0] = 1.0
    A[1, 1] = 2.0
    A[2, 2] = 3.0
    A.assemble()

    b = PETSc.Vec().createSeq(3)
    b.setArray([1.0, 2.0, 3.0])

    return PETScLinearSystemTask(params=None, A=A, b=b, x=None, rtol=1.0e-6, maxiter=100)


@pytest.fixture
def pytorch_task() -> PyTorchLinearSystemTask:
    A = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float64)
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    rtol = torch.tensor(1.0e-6)
    maxiter = torch.tensor(100)
    return PyTorchLinearSystemTask(params=None, A=A, b=b, x=None, rtol=rtol, maxiter=maxiter)


def test_generate_sample_pytorch_task():
    assert isinstance(generate_sample_pytorch_task(0), PyTorchLinearSystemTask)


def test_petsc2torch(petsc_task, pytorch_task):
    assert pytorch_task == petsc2torch(petsc_task)


def test_torch2petsc(pytorch_task, petsc_task):
    assert petsc_task == torch2petsc(pytorch_task)
