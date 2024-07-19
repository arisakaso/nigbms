# %%

import hydra
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sympy import IndexedBase, exp, lambdify, pi, sin, symbols
from tqdm import tqdm

from src.data.poisson1d import get_A

# %%


# %%
def get_symbolic_u_and_f(N_terms: int = 10):
    """get analytical "difficult" analytical u

    Args:
        num_terms (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    x = symbols("x")
    a = IndexedBase("a")
    u_sym = 0
    for i in range(1, N_terms + 1):
        u_sym += a[i] * sin(i * pi * x)
    f_sym = u_sym.diff(x, x)

    return u_sym, f_sym


def sample_random_coeffs(N_terms: int = 10, p: float = 0.5, type="quadratic", scale=1):
    """sample random coefficients for symbolic u

    Args:
        num_terms (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """

    is_difficult = np.random.binomial(1, p=p)
    if type == "quadratic":
        if is_difficult == 1:
            ais = np.random.normal(0, np.linspace(scale, 0, N_terms) ** 2, N_terms)
        else:
            ais = np.random.normal(0, np.linspace(0, scale, N_terms) ** 2, N_terms)

    elif type == "linear":
        if is_difficult == 1:
            ais = np.random.normal(0, np.linspace(scale, 0, N_terms), N_terms)
        else:
            ais = np.random.normal(0, np.linspace(0, scale, N_terms), N_terms)

    elif type == "constant":
        # ais = np.random.normal(0, scale, N_terms)
        ais = np.random.uniform(high=1, low=-1, size=N_terms)
    elif type == "default":
        if is_difficult == 1:
            ais = np.random.normal(
                0, scale * np.abs(np.linspace(-1, 1, N_terms)), N_terms
            )  # difficult
        else:
            ais = np.random.normal(
                0, scale * (1 - np.abs(np.linspace(-1, 1, N_terms))), N_terms
            )  # easy

    else:
        raise NotImplementedError

    return np.append(ais, is_difficult)


def get_symbolic_u_and_f_gaussian(N_terms: int = 10):
    """get analytical "difficult" analytical u

    Args:
        num_terms (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    x = symbols("x")
    a = IndexedBase("a")
    b = IndexedBase("b")
    c = IndexedBase("c")

    u_sym = 0
    for i in range(N_terms):
        u_sym += a[i] * (exp(-(((x - b[i]) / c[i]) ** 2)))
    f_sym = u_sym.diff(x, x)

    return f_sym, u_sym


def sample_random_coeffs_gaussian(N_terms=10):
    a = IndexedBase("a")
    b = IndexedBase("b")
    c = IndexedBase("c")

    ais = np.random.randn(N_terms)
    bis = np.random.uniform(0, 1, size=N_terms)
    cis = np.random.uniform(1e-2, 1, size=N_terms)

    subs = {
        **{a[i]: ais[i] for i in range(N_terms)},
        **{b[i]: bis[i] for i in range(N_terms)},
        **{c[i]: cis[i] for i in range(N_terms)},
    }

    return subs


def get_meta_df(N_data, N_terms: int = 10, p: float = 0.5, type="quadratic", scale=1):
    coeffs = [
        sample_random_coeffs(N_terms=N_terms, p=p, type=type, scale=scale)
        for _ in range(N_data)
    ]
    a = IndexedBase("a")
    meta_df = pd.DataFrame(
        coeffs, columns=[a[i] for i in range(1, N_terms + 1)] + ["is_difficult"]
    )
    return meta_df


def get_meta_df_gaussian(N_data, N_terms: int = 10):
    coeffs = [sample_random_coeffs_gaussian(N_terms=N_terms) for _ in range(N_data)]
    meta_df = pd.DataFrame(coeffs)
    return meta_df


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


def generate_poisson1d(u_sym, f_sym, i, coeffs, exact, N_grid):
    coeffs.pop("is_difficult", None)  # delete

    u_sym = u_sym.xreplace(coeffs)
    u = discretize(u_sym, N_grid + 2)
    u = torch.from_numpy(u)

    if exact:
        A_with_boundary = get_A(N_grid, with_boundary=True, dtype=torch.float64)
        f = A_with_boundary @ u
        # deal with boundary
        f = f[1:-1]
        f[0] -= u[0]
        f[-1] -= u[-1]
        u = u[1:-1]
    else:
        f_sym = f_sym.xreplace(coeffs)
        f = discretize(f_sym, N_grid + 2)
        f = torch.from_numpy(f)

    A = get_A(N_grid, dtype=torch.float64)
    assert torch.allclose(A @ u, f)

    torch.save([f, u], f"{i}_fu.pt")


# %%
@hydra.main(config_path="../configs", config_name="generate_poisson1d")
def main(cfg):
    if cfg.type == "gaussian":
        u_sym, f_sym = get_symbolic_u_and_f_gaussian(cfg.N_terms)
        meta_df = get_meta_df_gaussian(cfg.N_data, cfg.N_terms)
        meta_df.to_csv("meta_df.csv", index=False)
        Parallel(verbose=10, n_jobs=-1)(
            [
                delayed(generate_poisson1d)(
                    u_sym, f_sym, i, coeffs.to_dict(), cfg.exact, cfg.N_grid
                )
                for i, coeffs in tqdm(meta_df.iterrows(), total=len(meta_df))
            ]
        )
    else:
        u_sym, f_sym = get_symbolic_u_and_f(cfg.N_terms)
        meta_df = get_meta_df(cfg.N_data, cfg.N_terms, cfg.p, cfg.type, cfg.scale)
        meta_df.to_csv("meta_df.csv", index=False)
        Parallel(verbose=10, n_jobs=-1)(
            [
                delayed(generate_poisson1d)(
                    u_sym, f_sym, i, coeffs.to_dict(), cfg.exact, cfg.N_grid
                )
                for i, coeffs in tqdm(meta_df.iterrows(), total=len(meta_df))
            ]
        )


if __name__ == "__main__":
    main()
