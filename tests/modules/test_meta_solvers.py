import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from nigbms.modules.meta_solvers import ConstantMetaSolver
from nigbms.modules.tasks import Task


class TestPoisson1DMetaSolver:
    pass


class TestConstantMetaSolver:
    def test_forward(self):
        from nigbms.configs.modules.meta_solvers.configs import ConstantMetaSolverConfig

        with initialize(version_base="1.3", config_path="."):
            cfg: ConstantMetaSolverConfig = compose(config_name="constant_meta_solver_default")
        meta_solver = instantiate(cfg)
        tau = Task()
        theta = meta_solver(tau)
        assert isinstance(meta_solver, ConstantMetaSolver)
        assert theta.shape == torch.Size(cfg.model.shape)
