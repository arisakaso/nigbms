# %%
import hydra
import joblib
import numpy as np
import pandas as pd
import scipy
import torch
from fenics import *
from scipy.stats import loguniform
from tqdm import tqdm


# %%
def clamped_boundary(x, on_boundary):
    return on_boundary and x[0] < 1e-14


def epsilon(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)


def sigma(u, d, mu, lam):
    return lam * div(u) * Identity(d) + 2 * mu * epsilon(u)


def generate_elasticity_problem(L, W, mu, lam, rho):
    delta = W / L
    g = 0.4 * delta**2
    mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 3, 3, 3)
    V = VectorFunctionSpace(mesh, "P", 1)
    bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)
    u = TrialFunction(V)
    d = u.geometric_dimension()  # space dimension
    v = TestFunction(V)
    f = Constant((0, 0, -rho * g))
    T = Constant((0, 0, 0))
    a = inner(sigma(u, d, mu, lam), epsilon(v)) * dx
    L = dot(f, v) * dx + dot(T, v) * ds
    A, b = assemble_system(a, L, bc)
    A = A.array()
    b = b.get_local()
    x = scipy.linalg.solve(A, b)

    return A, b, x


def func(i, L, W, mu, lam, rho):
    A, b, x = generate_elasticity_problem(L, W, mu, lam, rho)
    A = torch.from_numpy(A)
    b = torch.from_numpy(b)
    x = torch.from_numpy(x)
    torch.save([b, x], f"{i}_fu.pt")
    torch.save(A, f"{i}_A.pt")


# %%
@hydra.main(config_path="../configs", config_name="generate_beam")
def main(cfg):
    Ls = np.ones(cfg.N_data)
    if cfg.W.const:
        Ws = np.ones(cfg.N_data) * cfg.W.const
    else:
        Ws = loguniform.rvs(cfg.W.min, cfg.W.max, size=cfg.N_data)
    if cfg.mu.const:
        mus = np.ones(cfg.N_data) * cfg.mu.const
    else:
        mus = loguniform.rvs(cfg.mu.min, cfg.mu.max, size=cfg.N_data)
    if cfg.lam.const:
        lams = np.ones(cfg.N_data) * cfg.lam.const
    else:
        lams = loguniform.rvs(cfg.lam.min, cfg.lam.max, size=cfg.N_data)
    rhos = loguniform.rvs(cfg.rho.min, cfg.rho.max, size=cfg.N_data)
    meta_df = pd.DataFrame({"L": Ls, "W": Ws, "mu": mus, "lam": lams, "rho": rhos})
    meta_df.to_csv("meta_df.csv", index=False)

    joblib.Parallel(n_jobs=-1)(
        joblib.delayed(func)(*params)
        for params in tqdm(meta_df.itertuples(), total=len(meta_df))
    )


if __name__ == "__main__":
    main()
