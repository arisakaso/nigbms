import shutil
from pathlib import Path

import pytest
from nigbms.tasks import (
    PETScLinearSystemTask,
    PyTorchLinearSystemTask,
    TaskConstructor,
    TaskParams,
    generate_sample_batched_petsc_task,
    generate_sample_batched_pytorch_task,
    generate_sample_petsc_task,
    generate_sample_pytorch_task,
    save_petsc_task,
    save_pytorch_task,
)


class TestPyTorchTaskConstructor(TaskConstructor):
    def __call__(self, params: TaskParams) -> PyTorchLinearSystemTask:
        return generate_sample_pytorch_task(params.seed)


class TestPETScTaskConstructor(TaskConstructor):
    def __call__(self, params: TaskParams) -> PETScLinearSystemTask:
        return generate_sample_petsc_task(params.seed)


@pytest.fixture(scope="session", autouse=True)
def temp_directory():
    dir_path = Path("/workspaces/nigbms/data/raw/temp")

    for i in range(10):
        save_pytorch_task(generate_sample_pytorch_task(i), dir_path / "pytorch" / str(i))
        save_petsc_task(generate_sample_petsc_task(i), dir_path / "petsc" / str(i))

    yield dir_path

    shutil.rmtree(dir_path)


@pytest.fixture(scope="session")
def pytorch_task_constructor() -> TestPyTorchTaskConstructor:
    return TestPyTorchTaskConstructor()


@pytest.fixture(scope="session")
def petsc_task_constructor() -> TestPETScTaskConstructor:
    return TestPETScTaskConstructor()


@pytest.fixture(scope="session")
def pytorch_task() -> PyTorchLinearSystemTask:
    return generate_sample_pytorch_task()


@pytest.fixture(scope="session")
def batched_pytorch_task() -> PyTorchLinearSystemTask:
    return generate_sample_batched_pytorch_task()


@pytest.fixture(scope="session")
def petsc_task() -> PETScLinearSystemTask:
    return generate_sample_petsc_task()


@pytest.fixture(scope="session")
def batched_petsc_task() -> PETScLinearSystemTask:
    return generate_sample_batched_petsc_task()