# %%
import shutil
from pathlib import Path

import torch

from nigbms.modules.tasks import (
    PETScLinearSystemTask,
    PyTorchLinearSystemTask,
    generate_sample_petsc_task,
    generate_sample_pytorch_task,
    load_petsc_task,
    load_pytorch_task,
    petsc2torch,
    save_petsc_task,
    save_pytorch_task,
    torch2petsc,
)


def test_generate_sample_pytorch_task():
    assert isinstance(generate_sample_pytorch_task(), PyTorchLinearSystemTask)


def test_generate_sample_petsc_task():
    assert isinstance(generate_sample_petsc_task(), PETScLinearSystemTask)


def test_petsc2torch():
    pytorch_task = generate_sample_pytorch_task()
    petsc_task = generate_sample_petsc_task()
    converted_task = petsc2torch(petsc_task)
    assert torch.equal(pytorch_task.A, converted_task.A)


def test_torch2petsc():
    pytorch_task = generate_sample_pytorch_task()
    petsc_task = generate_sample_petsc_task()
    converted_task = torch2petsc(pytorch_task)
    assert petsc_task.A.equal(converted_task.A)


def test_save_petsc_task():
    petsc_task = generate_sample_petsc_task()
    path = Path("tmp")
    save_petsc_task(petsc_task, path)
    task = load_petsc_task(path)
    assert isinstance(task, PETScLinearSystemTask)
    assert petsc_task.A.equal(task.A)
    assert petsc_task.b.equal(task.b)
    shutil.rmtree(path)


def test_save_pytorch_task():
    pytorch_task = generate_sample_pytorch_task()
    path = Path("tmp")
    save_pytorch_task(pytorch_task, path)
    task = load_pytorch_task(path)
    assert isinstance(task, PyTorchLinearSystemTask)
    assert torch.equal(pytorch_task.A, task.A)
    assert torch.equal(pytorch_task.b, task.b)
    shutil.rmtree(path)
