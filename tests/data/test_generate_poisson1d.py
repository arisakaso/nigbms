import numpy as np
from numpy.testing import assert_array_equal
from sympy import pi, sin, symbols

from nigbms.data.generate_poisson1d import (
    CoefsDistribution,
    Poisson1DParams,
    construct_pytorch_poisson1d_task,
    construct_sym_u,
    discretize,
    laplacian_matrix,
)
from nigbms.modules.tasks import PyTorchLinearSystemTask


def test_laplacian_matrix() -> None:
    expected_result = np.array(
        [
            [2, -1, 0, 0, 0],
            [-1, 2, -1, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 0, -1, 2, -1],
            [0, 0, 0, -1, 2],
        ]
    )
    assert_array_equal(laplacian_matrix(5), expected_result)


def test_discretize() -> None:
    x = symbols("x")
    expr = sin(x)
    N_grid = 128

    expected_result = np.sin(np.linspace(0, 1, N_grid))
    result = discretize(expr, N_grid)

    assert_array_equal(result, expected_result)


def test_construct_sym_u() -> None:
    N_terms = 3
    coefs = np.ones(3)
    u_sym = construct_sym_u(N_terms, coefs)
    assert u_sym == sum([coefs[i] * sin((i + 1) * pi * symbols("x")) for i in range(N_terms)])


def test_construct_pytorch_poisson1d_task() -> None:
    parasm = Poisson1DParams()
    task = construct_pytorch_poisson1d_task(parasm)
    assert isinstance(task, PyTorchLinearSystemTask)


class TestCoefsDistribution:
    def test_sample_multiple(self) -> None:
        dist = CoefsDistribution([10], 0.1, 1)
        coefs = dist.sample(0)
        assert isinstance(coefs, np.ndarray)
        assert coefs.shape == (10,)
