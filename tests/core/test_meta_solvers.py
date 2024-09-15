import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from nigbms.configs.meta_solvers import ConstantMetaSolverConfig, Poisson1DMetaSolverConfig
from nigbms.meta_solvers import ConstantMetaSolver, Poisson1DMetaSolver
from nigbms.tasks import Task, generate_sample_batched_pytorch_task


class TestConstantMetaSolver:
    def test_forward(self):
        with initialize(version_base="1.3"):
            cfg: ConstantMetaSolverConfig = compose(overrides=["+meta_solver@_global_=constant_meta_solver_default"])
        meta_solver = instantiate(cfg)
        assert isinstance(meta_solver, ConstantMetaSolver)

        tau = Task()
        theta = meta_solver(tau)
        assert theta.shape == torch.Size(cfg.model.shape)


class TestPoisson1DMetaSolver:
    def test_forward(self):
        with initialize(version_base="1.3"):
            cfg: Poisson1DMetaSolverConfig = compose(overrides=["+meta_solver@_global_=poisson1d_meta_solver_default"])
        torch.set_default_dtype(torch.float64)
        meta_solver = instantiate(cfg)
        assert isinstance(meta_solver, Poisson1DMetaSolver)

        tau = generate_sample_batched_pytorch_task()
        theta = meta_solver(tau)
        assert theta.shape == torch.Size([len(tau), cfg.model.out_dim])
