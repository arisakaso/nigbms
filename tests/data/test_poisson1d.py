import numpy as np

from nigbms.data.generate_poisson1d import laplacian_matrix, sample_random_coeffs


def test_laplacian_matrix():
    N = 5
    expected_result = np.array(
        [[2, -1, 0, 0, 0], [-1, 2, -1, 0, 0], [0, -1, 2, -1, 0], [0, 0, -1, 2, -1], [0, 0, 0, -1, 2]]
    )
    assert np.array_equal(laplacian_matrix(N), expected_result)


def test_sample_random_coeffs():
    # # Test case 1: Quadratic coefficients, easy difficulty
    # np.random.seed(0)
    # N_terms = 5
    # p = 0.2
    # type = "quadratic"
    # scale = 1
    # expected_result = np.array(
    #     [-0.00000000e00, -1.00000000e00, -2.00000000e00, -3.00000000e00, -4.00000000e00, 0.00000000e00]
    # )
    # assert np.array_equal(sample_random_coeffs(N_terms, p, type, scale), expected_result)

    # # Test case 2: Constant coefficients
    # np.random.seed(1)
    # N_terms = 3
    # p = 0.5
    # type = "constant"
    # scale = 2
    # expected_result = np.array([-0.73127151, -0.02455124, 0.69486724, 1.00000000])
    # assert np.array_equal(sample_random_coeffs(N_terms, p, type, scale), expected_result)

    # # Test case 3: Linear coefficients, difficult difficulty
    # np.random.seed(2)
    # N_terms = 4
    # p = 0.8
    # type = "linear"
    # scale = 0.5
    # expected_result = np.array([-0.00000000e00, -1.00000000e-01, -2.00000000e-01, -3.00000000e-01, 1.00000000e00])
    # assert np.array_equal(sample_random_coeffs(N_terms, p, type, scale), expected_result)

    # Test case 4: Invalid type
    N_terms = 2
    p = 0.3
    type = "invalid"
    scale = 1
    try:
        sample_random_coeffs(N_terms, p, type, scale)
        raise AssertionError("Expected ValueError")
    except ValueError:
        assert True
