import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from nigbms.modules.data import pytorch_task_collate_fn
from nigbms.modules.solvers import PyTorchCG, PyTorchJacobi, PyTorchSOR
from nigbms.modules.tasks import PyTorchLinearSystemTask


@pytest.fixture
def batched_pytorch_task():
    tasks = []
    for i in range(5):
        torch.manual_seed(i)
        params = torch.randn(5)
        root_A = torch.randn(5, 5, dtype=torch.float64)
        A = root_A @ root_A.T + 10 * torch.eye(5)  # SPD and diagonally dominant
        x = torch.ones(5, 1, dtype=torch.float64)
        b = A @ x
        rtol = torch.tensor(1.0e-6)
        maxiter = torch.tensor(100)
        tasks.append(PyTorchLinearSystemTask(params=params, A=A, b=b, x=x, rtol=rtol, maxiter=maxiter))
    batched_tasks = pytorch_task_collate_fn(tasks)
    return batched_tasks


def test_pytorch_jacobi(batched_pytorch_task):
    solver = PyTorchJacobi(params_fix=OmegaConf.create({"history_length": 100}), params_learn={})
    theta = TensorDict({"x0": torch.zeros_like(batched_pytorch_task.x)})
    solver.forward(batched_pytorch_task, theta)
    assert torch.allclose(solver.x, batched_pytorch_task.x)


def test_pytorch_sor(batched_pytorch_task):
    solver = PyTorchSOR(params_fix=OmegaConf.create({"history_length": 100, "omega": 1.0}), params_learn={})
    theta = TensorDict({"x0": torch.zeros_like(batched_pytorch_task.x)})
    solver.forward(batched_pytorch_task, theta)
    assert torch.allclose(solver.x, batched_pytorch_task.x)


def test_pytorch_cg(batched_pytorch_task):
    solver = PyTorchCG(params_fix=OmegaConf.create({"history_length": 100}), params_learn={})
    theta = TensorDict({"x0": torch.zeros_like(batched_pytorch_task.x)})
    solver.forward(batched_pytorch_task, theta)
    assert torch.allclose(solver.x, batched_pytorch_task.x)
