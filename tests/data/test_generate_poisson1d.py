import numpy as np
from numpy.testing import assert_array_equal
from sympy import sin, symbols

from nigbms.data.generate_poisson1d import discretize, laplacian_matrix, sample_random_coeffs


def test_laplacian_matrix():
    N = 5
    expected_result = np.array(
        [[2, -1, 0, 0, 0], [-1, 2, -1, 0, 0], [0, -1, 2, -1, 0], [0, 0, -1, 2, -1], [0, 0, 0, -1, 2]]
    )

    assert np.array_equal(laplacian_matrix(N), expected_result)


def test_sample_random_coeffs():
    N_terms = 2
    p = 0.3
    type = "invalid"
    scale = 1

    try:
        sample_random_coeffs(N_terms, p, type, scale)
        raise AssertionError("Expected ValueError")
    except ValueError:
        assert True


def test_discretize():
    x = symbols("x")
    expr = sin(x)
    N_grid = 128

    expected_result = np.sin(np.linspace(0, 1, N_grid))
    result = discretize(expr, N_grid)

    assert_array_equal(result, expected_result)
