# %%

import hydra
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.sparse import diags
from sympy import IndexedBase, lambdify, pi, sin, symbols
from tqdm import tqdm

x = symbols("x")
a = IndexedBase("a")


# %%
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


def get_sympy_u(N_terms: int = 10):
    """Calculate the symbolic representation of u.

    This function calculates the symbolic representation of u based on the given number of terms.

    Args:
        N_terms (int, optional): The number of terms to consider in the calculation. Defaults to 10.

    Returns:
        _type_: The symbolic representation of u.
    """
    u_sym = 0
    for i in range(1, N_terms + 1):
        u_sym += a[i] * sin(i * pi * x)
    return u_sym


def get_sympy_f(u_sym):
    """
    Calculate the symbolic representation of f.

    Parameters:
        u_sym (sympy.Symbol): The symbolic representation of u.

    Returns:
        sympy.Symbol: The symbolic representation of f.
    """

    return u_sym.diff(x, x)


def discretize(expr, N_grid=128):
    """discretize sympy expression in [0, 1]

    Args:
        expr ([type]): [description]
        N (int, optional): [description]. Defaults to 128.

    Returns:
        [type]: [description]
    """
    expr = lambdify(symbols("x"), expr, "numpy")
    return expr(np.linspace(0, 1, N_grid))


def sample_random_coeffs(N_terms: int = 10, p: float = 0.5, type="quadratic", scale=1):
    """
    Generate random coefficients for a given number of terms.

    Args:
        N_terms (int): Number of terms to generate coefficients for. Default is 10.
        p (float): Probability of generating difficult coefficients. Default is 0.5.
        type (str): Type of coefficients to generate. Options are "quadratic" (default) and "constant".
        scale (float): Scaling factor for the coefficients. Default is 1.

    Returns:
        numpy.ndarray: Array of generated coefficients, with an additional element indicating difficulty.
    """

    is_difficult = np.random.binomial(1, p=p)

    if type == "quadratic":
        if is_difficult == 1:
            ais = np.random.normal(0, scale * np.linspace(-1, 1, N_terms) ** 2, N_terms)  # difficult
        else:
            ais = np.random.normal(0, scale * (1 - np.linspace(-1, 1, N_terms) ** 2), N_terms)  # easy

    elif type == "constant":
        ais = np.random.uniform(high=1, low=-1, size=N_terms)

    elif type == "linear":
        if is_difficult == 1:
            ais = np.random.normal(0, scale * np.abs(np.linspace(-1, 1, N_terms)), N_terms)  # difficult
        else:
            ais = np.random.normal(0, scale * (1 - np.abs(np.linspace(-1, 1, N_terms))), N_terms)  # easy

    else:
        raise ValueError("Invalid type")

    return np.append(ais, is_difficult)


def get_meta_df(N_data, N_terms: int = 10, p: float = 0.5, type="quadratic", scale=1):
    coeffs = [sample_random_coeffs(N_terms=N_terms, p=p, type=type, scale=scale) for _ in range(N_data)]
    column_names = [a[i] for i in range(1, N_terms + 1)] + ["is_difficult"]
    meta_df = pd.DataFrame(coeffs, columns=column_names)
    return meta_df


def generate_poisson1d(u_sym, i, coeffs, exact):
    coeffs.pop("is_difficult")  # delete
    N_terms = len(coeffs)

    u_sym = u_sym.xreplace(coeffs)
    u = discretize(u_sym, N_terms + 2)
    u = torch.from_numpy(u)

    if exact:
        A = laplacian_matrix(N_terms, with_boundary=True).double()
        f = A @ u
        # deal with boundary
        f = f[1:-1]
        f[0] -= u[0]
        f[-1] -= u[-1]
        u = u[1:-1]
    else:
        f_sym = get_sympy_f(u_sym)
        f = discretize(f_sym, N_terms + 2)
        f = torch.from_numpy(f)

    A = laplacian_matrix(N_terms).double()
    assert torch.allclose(A @ u, f)

    np.save(f"{i}_u.npy", u.numpy())
    np.save(f"{i}_f.npy", f.numpy())


# %%
@hydra.main(config_path="../configs", config_name="generate_poisson1d", version_base=None)
def main(cfg):
    u_sym = get_sympy_u(N_terms=cfg.N_grid)
    meta_df = get_meta_df(cfg.N_data, cfg.N_grid, cfg.p, cfg.type, cfg.scale)
    meta_df.to_csv("meta_df.csv", index=False)
    Parallel(verbose=10, n_jobs=-1)(
        [
            delayed(generate_poisson1d)(u_sym, i, coeffs.to_dict(), exact=cfg.exact)
            for i, coeffs in tqdm(meta_df.iterrows(), total=len(meta_df))
        ]
    )


if __name__ == "__main__":
    main()
