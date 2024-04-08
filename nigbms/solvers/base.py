from tensordict import TensorDict
from torch.nn import Module


class _Solver(Module):
    """Base class for solvers."""

    def __init__(self, params_fix: dict, params_learn: dict) -> None:
        super().__init__()
        self.params_fix = params_fix
        self.params_learn = params_learn

    def _setup(self, tau: dict, theta: TensorDict):
        raise NotImplementedError

    def forward(self, tau: dict, theta: TensorDict):
        raise NotImplementedError
