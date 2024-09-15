import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
from nigbms.wrapper import WrappedSolver


class TestWrappedSolver:
    @pytest.fixture
    def init_wrapper(self):
        with initialize(version_base="1.3", config_path="../configs/train"):
            cfg = compose(config_name="minimize_testfunctions")
            cfg.wandb.project = "test"
            solver = instantiate(cfg.solver)
            surrogate = instantiate(cfg.surrogate)
            constructor = instantiate(cfg.constructor)
            self.wrapper = instantiate(cfg.wrapper, solver=solver, surrogate=surrogate, constructor=constructor)

    def test_init(self, init_wrapper):
        assert isinstance(self.wrapper, WrappedSolver)
