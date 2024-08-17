# %%
import shutil
from pathlib import Path

import pytest
import torch
from petsc4py import PETSc
from tensordict import TensorDict

from nigbms.modules.tasks import (
    PETScLinearSystemTask,
    PyTorchLinearSystemTask,
    generate_sample_pytorch_task,
    load_petsc_task,
    load_pytorch_task,
    petsc2torch,
    save_petsc_task,
    save_pytorch_task,
    torch2petsc,
)


@pytest.fixture
def petsc_task() -> PETScLinearSystemTask:
    params = TensorDict({"param1": torch.tensor(1.0), "param2": torch.ones(3)})
    A = PETSc.Mat().createAIJ([3, 3])
    A.setUp()
    A[0, 0] = 1.0
    A[1, 1] = 2.0
    A[2, 2] = 3.0
    A.assemble()

    b = PETSc.Vec().createSeq(3)
    b.setArray([1.0, 2.0, 3.0])

    return PETScLinearSystemTask(params=params, A=A, b=b, x=None, rtol=1.0e-6, maxiter=100)


@pytest.fixture
def pytorch_task() -> PyTorchLinearSystemTask:
    params = TensorDict({"param1": torch.tensor(1.0), "param2": torch.ones(3)})
    A = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=torch.float64)
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    rtol = torch.tensor(1.0e-6)
    maxiter = torch.tensor(100)
    return PyTorchLinearSystemTask(params=params, A=A, b=b, x=None, rtol=rtol, maxiter=maxiter)


def test_generate_sample_pytorch_task():
    assert isinstance(generate_sample_pytorch_task(0), PyTorchLinearSystemTask)


def test_petsc2torch(petsc_task, pytorch_task):
    converted_task = petsc2torch(petsc_task)
    assert torch.equal(pytorch_task.A, converted_task.A)


def test_torch2petsc(pytorch_task, petsc_task):
    converted_task = torch2petsc(pytorch_task)
    assert petsc_task.A.equal(converted_task.A)


def test_save_petsc_task(petsc_task):
    path = Path("tmp")
    save_petsc_task(petsc_task, path)
    task = load_petsc_task(path)
    assert isinstance(task, PETScLinearSystemTask)
    assert petsc_task.A.equal(task.A)
    assert petsc_task.b.equal(task.b)
    shutil.rmtree(path)


def test_save_pytorch_task(pytorch_task):
    path = Path("tmp")
    save_pytorch_task(pytorch_task, path)
    task = load_pytorch_task(path)
    assert isinstance(task, PyTorchLinearSystemTask)
    assert torch.equal(pytorch_task.A, task.A)
    assert torch.equal(pytorch_task.b, task.b)
    shutil.rmtree(path)
