import hydra
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import diags
from sympy import Expr, IndexedBase, lambdify, pi, sin, symbols
from tqdm import tqdm

x = symbols("x")
a = IndexedBase("a")


def laplacian_matrix(N: int, out_type="numpy"):
    """
    Generate a Laplacian matrix of size N.

    Parameters:
    - N (int): The size of the Laplacian matrix.
    - out_type (str): The output type of the matrix. Default is "numpy".

    Returns:
    - numpy.ndarray or scipy.sparse.spmatrix: The Laplacian matrix.

    """

    D = diags([-1, 2, -1], [-1, 0, 1], shape=(N, N))

    if out_type == "numpy":
        return D.toarray()

    else:
        return D


def get_sympy_u(N_terms: int) -> Expr:
    """Calculate the symbolic representation of u.

    This function calculates the symbolic representation of u based on the given number of terms.

    Args:
        N_terms (int, optional): The number of terms to consider in the calculation. Defaults to 10.

    Returns:
        Expr: The symbolic representation of u.
    """
    u_sym = 0
    for i in range(1, N_terms + 1):
        u_sym += a[i] * sin(i * pi * x)
    return u_sym


def get_sympy_f(u_sym: Expr) -> Expr:
    """
    Calculate the symbolic representation of f.

    Parameters:
        u_sym (sympy.Symbol): The symbolic representation of u.

    Returns:
        Expr: The symbolic representation of f.
    """

    return u_sym.diff(x, x)


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


def sample_random_coeffs(N_terms: int, p: float, distribution: str, scale: float = 1.0) -> np.ndarray:
    """
    Generate random coefficients for a given number of terms.

    Args:
        N_terms (int): Number of terms to generate coefficients for. Default is 10.
        p (float): Probability of generating difficult coefficients. Default is 0.5.
        distribution (str): distribution of coefficients to generate. Options are "quadratic" (default) and "constant".
        scale (float): Scaling factor for the coefficients. Default is 1.

    Returns:
        numpy.ndarray: Array of generated coefficients, with an additional element indicating difficulty.
    """

    is_difficult = np.random.binomial(1, p=p)

    if distribution == "quadratic":
        if is_difficult == 1:
            ais = np.random.normal(0, scale * np.linspace(-1, 1, N_terms) ** 2, N_terms)  # difficult
        else:
            ais = np.random.normal(0, scale * (1 - np.linspace(-1, 1, N_terms) ** 2), N_terms)  # easy

    elif distribution == "constant":
        ais = np.random.normal(0, scale, N_terms)

    elif distribution == "linear":
        if is_difficult == 1:
            ais = np.random.normal(0, scale * np.abs(np.linspace(-1, 1, N_terms)), N_terms)  # difficult
        else:
            ais = np.random.normal(0, scale * (1 - np.abs(np.linspace(-1, 1, N_terms))), N_terms)  # easy

    else:
        raise ValueError("Invalid type")

    return np.append(ais, is_difficult)


def get_meta_df(N_data: int, N_terms: int, p: float, distribution: str, scale: float = 1.0) -> pd.DataFrame:
    """
    Generate a meta dataframe with random coefficients.

    Parameters:
    - N_data (int): Number of data points to generate.
    - N_terms (int): Number of terms in each data point.
    - p (float): Probability of a term being difficult in a data point.
    - distribution (str): Type of distribution for generating coefficients.
    - scale (float, optional): Scaling factor for the coefficients. Default is 1.0.

    Returns:
    - meta_df (pandas.DataFrame): Meta dataframe with randomly generated coefficients.
    """
    coeffs = [sample_random_coeffs(N_terms, p, distribution, scale) for _ in range(N_data)]
    column_names = [a[i] for i in range(1, N_terms + 1)] + ["is_difficult"]
    meta_df = pd.DataFrame(coeffs, columns=column_names)
    return meta_df


def generate_poisson1d(A: np.ndarray, u_sym: Expr, i: int, coeffs: np.ndarray) -> None:
    """
    Generate 1D Poisson equation data.

    Args:
        u_sym (Expr): The symbolic expression representing the function u(x).
        i (int): The index of the data.
        coeffs (np.ndarray): The coefficients to substitute in the symbolic expression.

    Returns:
        None
    """
    coeffs.pop("is_difficult")  # delete
    N_terms = len(coeffs)

    u_sym = u_sym.xreplace(coeffs)
    x = discretize(u_sym, N_terms + 2)[1:-1]  # exclude boundary because it is fixed to 0
    b = A @ x
    assert np.allclose(A @ x, b)

    np.save(f"{i}_x.npy", x)
    np.save(f"{i}_b.npy", b)

    return None


@hydra.main(version_base="1.3", config_path="../configs/data", config_name="generate_poisson1d")
def main(cfg):
    u_sym = get_sympy_u(N_terms=cfg.N_terms)
    meta_df = get_meta_df(cfg.N_data, cfg.N_grid, cfg.p, cfg.distribution, cfg.scale)
    meta_df.to_csv("meta_df.csv", index=False)
    A = laplacian_matrix(cfg.N_grid)
    np.save("A.npy", A)
    Parallel(verbose=10, n_jobs=-1)(
        [
            delayed(generate_poisson1d)(A, u_sym, i, coeffs.to_dict())
            for i, coeffs in tqdm(meta_df.iterrows(), total=len(meta_df))
        ]
    )


if __name__ == "__main__":
    main()
