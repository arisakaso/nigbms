import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
from nigbms.configs import constructors, solvers, surrogates, wrapper  # noqa
from nigbms.wrapper import WrappedSolver


class TestWrappedSolver:
    @pytest.fixture
    def init_wrapper(self):
        with initialize(version_base="1.3"):
            solver = instantiate(compose(overrides=["+solver@_global_=testfunction_solver_default"]))
            surrogate = instantiate(compose(overrides=["+surrogate@_global_=testfunction_surrogate_default"]))
            constructor = instantiate(compose(overrides=["+constructor@_global_=constructor_default"]))
            self.wrapper = instantiate(
                compose(overrides=["+wrapper@_global_=wrapped_solver_default"]),
                solver=solver,
                surrogate=surrogate,
                constructor=constructor,
            )

    def test_init(self, init_wrapper):
        assert isinstance(self.wrapper, WrappedSolver)
