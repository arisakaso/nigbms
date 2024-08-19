import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from nigbms.configs.modules.meta_solvers.configs import ConstantMetaSolverConfig, Poisson1DMetaSolverConfig
from nigbms.modules.meta_solvers import ConstantMetaSolver, Poisson1DMetaSolver
from nigbms.modules.tasks import Task, generate_sample_batched_pytorch_task


class TestConstantMetaSolver:
    def test_forward(self):
        with initialize(version_base="1.3", config_path="."):
            cfg: ConstantMetaSolverConfig = compose(config_name="constant_meta_solver_default")
        meta_solver = instantiate(cfg)
        assert isinstance(meta_solver, ConstantMetaSolver)

        tau = Task()
        theta = meta_solver(tau)
        assert theta.dtype == torch.float64
        assert theta.shape == torch.Size(cfg.model.shape)


class TestPoisson1DMetaSolver:
    def test_forward(self):
        with initialize(version_base="1.3", config_path="."):
            cfg: Poisson1DMetaSolverConfig = compose(config_name="poisson1d_meta_solver_default")
        meta_solver = instantiate(cfg)
        assert isinstance(meta_solver, Poisson1DMetaSolver)

        tau = generate_sample_batched_pytorch_task()
        theta = meta_solver(tau)
        assert theta.dtype == torch.float64
        assert theta.shape == torch.Size([len(tau), cfg.model.out_dim])
