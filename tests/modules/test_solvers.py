from typing import List

import numpy as np
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tensordict import TensorDict

from nigbms.modules.data import pytorch_task_collate_fn
from nigbms.modules.solvers import PETScKSP, PyTorchCG, PyTorchJacobi, PyTorchSOR
from nigbms.modules.tasks import (
    PETScLinearSystemTask,
    PyTorchLinearSystemTask,
    generate_sample_pytorch_task,
    torch2petsc,
)


@pytest.fixture
def batched_pytorch_tasks() -> PyTorchLinearSystemTask:
    pytorch_tasks = [generate_sample_pytorch_task(seed) for seed in range(3)]
    batched_tasks = pytorch_task_collate_fn(pytorch_tasks)
    return batched_tasks


@pytest.fixture
def batched_petsc_tasks() -> List[PETScLinearSystemTask]:
    pytorch_tasks = [generate_sample_pytorch_task(seed) for seed in range(3)]
    batched_petsc_tasks = [torch2petsc(task) for task in pytorch_tasks]
    return batched_petsc_tasks


def test_pytorch_jacobi(batched_pytorch_tasks):
    solver = PyTorchJacobi(params_fix=OmegaConf.create({"history_length": 100}), params_learn={})
    theta = TensorDict({"x0": torch.zeros_like(batched_pytorch_tasks.x)})
    solver.forward(batched_pytorch_tasks, theta)
    assert torch.allclose(solver.x, batched_pytorch_tasks.x)


def test_pytorch_sor(batched_pytorch_tasks):
    solver = PyTorchSOR(params_fix=OmegaConf.create({"history_length": 100, "omega": 1.0}), params_learn={})
    theta = TensorDict({"x0": torch.zeros_like(batched_pytorch_tasks.x)})
    solver.forward(batched_pytorch_tasks, theta)
    assert torch.allclose(solver.x, batched_pytorch_tasks.x)


def test_pytorch_cg(batched_pytorch_tasks):
    solver = PyTorchCG(params_fix=OmegaConf.create({"history_length": 100}), params_learn={})
    theta = TensorDict({"x0": torch.zeros_like(batched_pytorch_tasks.x)})
    solver.forward(batched_pytorch_tasks, theta)
    assert torch.allclose(solver.x, batched_pytorch_tasks.x)


def test_petsc_default(batched_petsc_tasks):
    solver = PETScKSP(params_fix=OmegaConf.create({"history_length": 100}), params_learn={})
    theta = TensorDict({}, batch_size=3)
    solver.forward(batched_petsc_tasks, theta)
    assert all([np.allclose(solver.x[i].getArray(), batched_petsc_tasks[i].x.getArray()) for i in range(3)])


def test_petsc_cg(batched_petsc_tasks):
    with initialize(version_base="1.3", config_path="../configs/modules"):
        cfg = compose(config_name="solvers")
    solver = instantiate(cfg.petsc_cg)
    theta = TensorDict({}, batch_size=3)
    solver.forward(batched_petsc_tasks, theta)
    assert all([np.allclose(solver.x[i].getArray(), batched_petsc_tasks[i].x.getArray()) for i in range(3)])
