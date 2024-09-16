import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from nigbms.configs.solvers import (
    PETScCGConfig,
    PETScJacobiConfig,
    PETScKSPConfig,
    PyTorchJacobiConfig,
    TestFunctionConfig,
)
from nigbms.solvers import PETScKSP, PyTorchCG, PyTorchJacobi, PyTorchSOR
from nigbms.tasks import (
    MinimizeTestFunctionTask,
)
from omegaconf import OmegaConf
from tensordict import TensorDict


def test_test_function_solver():
    with initialize(version_base="1.3"):
        cfg: TestFunctionConfig = compose(overrides=["+solver@_global_=testfunction_solver_default"])
    solver = instantiate(cfg)
    tau = MinimizeTestFunctionTask(f=lambda x: torch.sum(x**2))
    theta = TensorDict({"x": torch.zeros(10)})
    assert torch.allclose(solver(tau, theta), theta["x"])


def test_pytorch_jacobi(batched_pytorch_task):
    with initialize(version_base="1.3"):
        cfg: PyTorchJacobiConfig = compose(overrides=["+solver@_global_=pytorch_jacobi_default"])
    solver = instantiate(cfg)
    theta = TensorDict({"x0": torch.zeros_like(batched_pytorch_task.x)})
    assert isinstance(solver, PyTorchJacobi)

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


def test_petsc_default(batched_petsc_task):
    solver = PETScKSP(params_fix=OmegaConf.create({"history_length": 100}), params_learn={})
    theta = TensorDict({}, batch_size=3)
    solver.forward(batched_petsc_task, theta)
    assert all([np.allclose(solver.x[i].getArray(), batched_petsc_task.get_task(i).x.getArray()) for i in range(3)])


def test_petsc_ksp_default(batched_petsc_task):
    with initialize(version_base="1.3"):
        cfg: PETScKSPConfig = compose(overrides=["+solver@_global_=petsc_ksp_default"])
    solver = instantiate(cfg)
    theta = TensorDict({}, batch_size=3)
    solver.forward(batched_petsc_task, theta)
    assert all([np.allclose(solver.x[i].getArray(), batched_petsc_task.get_task(i).x.getArray()) for i in range(3)])


def test_petsc_cg(batched_petsc_task):
    with initialize(version_base="1.3"):
        cfg: PETScCGConfig = compose(overrides=["+solver@_global_=petsc_cg_default"])

    assert cfg.params_fix.ksp_type == "cg"
    solver = instantiate(cfg)
    theta = TensorDict({}, batch_size=3)
    solver.forward(batched_petsc_task, theta)
    assert all([np.allclose(solver.x[i].getArray(), batched_petsc_task.get_task(i).x.getArray()) for i in range(3)])


def test_petsc_jacobi(batched_petsc_task):
    with initialize(version_base="1.3"):
        cfg: PETScJacobiConfig = compose(overrides=["+solver@_global_=petsc_jacobi_default"])

    assert cfg.params_fix.ksp_type == "richardson"
    assert cfg.params_fix.pc_type == "jacobi"
    solver = instantiate(cfg)
    theta = TensorDict({}, batch_size=3)
    solver.forward(batched_petsc_task, theta)
    assert all([np.allclose(solver.x[i].getArray(), batched_petsc_task.get_task(i).x.getArray()) for i in range(3)])


def test_equivalence_of_pytorch_and_petsc(batched_pytorch_task, batched_petsc_task):
    torch.set_default_dtype(torch.float64)
    with initialize(version_base="1.3"):
        cfg: PyTorchJacobiConfig = compose(overrides=["+solver@_global_=pytorch_jacobi_default"])
    pytorch_solver = instantiate(cfg)
    theta = TensorDict({}, batch_size=3)
    pytorch_hist = pytorch_solver.forward(batched_pytorch_task, theta)

    with initialize(version_base="1.3"):
        cfg: PETScJacobiConfig = compose(overrides=["+solver@_global_=petsc_jacobi_default"])
    petsc_solver = instantiate(cfg)
    theta = TensorDict({}, batch_size=3)
    petsc_hist = petsc_solver.forward(batched_petsc_task, theta)

    assert all([np.allclose(pytorch_solver.x, petsc_solver.x[i].getArray()) for i in range(3)])
    assert torch.allclose(pytorch_hist, petsc_hist)


# def test_pytorch_jitjacobi(batched_pytorch_task):
#     with initialize(version_base="1.3"):
#         cfg: PyTorchJacobiConfig = compose(overrides=["+solver@_global_=pytorch_jacobi_default"])
#         cfg._target_ = "nigbms.modules.solvers.JITJacobi"
#         cfg.params_learn = {}
#         cfg.params_fix = OmegaConf.create({"history_length": 100})
#     solver = torch.jit.script(instantiate(cfg))
#     theta = TensorDict({"x0": torch.zeros_like(batched_pytorch_task.x)})
#     assert isinstance(solver, JITJacobi)

#     solver.forward(batched_pytorch_task, theta)
#     assert torch.allclose(solver.x, batched_pytorch_task.x)
