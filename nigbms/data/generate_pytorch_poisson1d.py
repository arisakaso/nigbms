from dataclasses import asdict, dataclass

import numpy as np
from sympy import Expr, S, lambdify, pi, sin, symbols
from tensordict import TensorDict
from torch import tensor

from nigbms.modules.tasks import PyTorchLinearSystemTask, TaskParams
from nigbms.utils.distributions import Distribution


@dataclass
class Poisson1DParams(TaskParams):
    N_terms: int = 10
    N_grid: int = 100
    coefs: np.ndarray = np.ones(10)
    rtol: float = 1e-6
    maxiter: int = 100


class CoefsDistribution(Distribution):
    def __init__(self, shape, p: float, scale: float):
        assert len(shape) == 1
        super().__init__(shape)
        self.p = p
        self.scale = scale

    def sample(self, seed: int = None) -> np.ndarray:
        np.random.seed(seed)
        is_difficult = np.random.choice([True, False], p=[self.p, 1 - self.p])
        if is_difficult:
            return np.random.normal(0, self.scale * np.linspace(-1, 1, self.shape[0]) ** 2, self.shape[0])
        else:
            return np.random.normal(0, self.scale * (1 - np.abs(np.linspace(-1, 1, self.shape[0]))), self.shape[0])


def laplacian_matrix(N: int) -> np.ndarray:
    """
    Generate a Laplacian matrix of size N.

    Parameters:
    - N (int): The size of the Laplacian matrix.
    - out_type (str): The output type of the matrix. Default is "numpy".

    Returns:
    - numpy.ndarray or scipy.sparse.spmatrix: The Laplacian matrix.

    """

    D = np.zeros((N, N))
    D += np.diag([-1] * (N - 1), k=-1)
    D += np.diag([2] * N, k=0)
    D += np.diag([-1] * (N - 1), k=1)
    return D


def discretize(expr: Expr, N_grid: int) -> np.ndarray:
    """
    Discretizes the given expression in [0, 1] over a specified number of grid points.
    This includes the values at the boundaries.

    Parameters:
    expr (sympy.Expr): The expression to be discretized.
    N_grid (int): The number of grid points.

    Returns:
    numpy.ndarray: An array containing the discretized values of the expression.
    """
    lambdified_expr = lambdify(symbols("x"), expr, "numpy")
    return lambdified_expr(np.linspace(0, 1, N_grid))


def construct_sym_u(N_terms: int, coefs: np.ndarray) -> Expr:
    x = symbols("x")
    u_sym = S(0)
    for i in range(N_terms):
        u_sym += coefs[i] * sin((i + 1) * pi * x)
    return u_sym


def construct_pytorch_poisson1d_task(params: Poisson1DParams) -> PyTorchLinearSystemTask:
    u_sym = construct_sym_u(params.N_terms, params.coefs)
    x = discretize(u_sym, params.N_grid + 2)[1:-1]  # exclude boundary because it is fixed to 0
    A = laplacian_matrix(params.N_grid)
    b = A @ x
    task = PyTorchLinearSystemTask(
        TensorDict(asdict(params)),
        tensor(A),
        tensor(b),
        tensor(x),
        tensor(params.rtol),
        tensor(params.maxiter),
    )
    return task
