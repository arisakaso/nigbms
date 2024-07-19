# %%
import hydra
import joblib
import numpy as np
import pandas as pd
import torch
from fenics import *
from scipy.stats import loguniform
from tqdm import tqdm

# %%
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["ghost_mode"] = "shared_facet"


def generate_biharmonic_problem(c1, c2, c3):
    # Optimization options for the form compiler

    mesh = UnitSquareMesh(3, 3)
    V = FunctionSpace(mesh, "CG", 2)

    # Define Dirichlet boundary
    class DirichletBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class Source(UserExpression):
        def eval(self, values, x):
            values[0] = c1 * sin(c2 * pi * x[0]) * sin(c3 * pi * x[1])

    # Define boundary condition
    u0 = Constant(0.0)
    bc = DirichletBC(V, u0, DirichletBoundary())

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)

    # Define normal component, mesh size and right-hand side
    h = CellDiameter(mesh)
    h_avg = (h("+") + h("-")) / 2.0
    n = FacetNormal(mesh)
    f = Source(degree=2)

    # Penalty parameter
    alpha = Constant(8.0)

    # Define bilinear form
    a = (
        inner(div(grad(u)), div(grad(v))) * dx
        - inner(avg(div(grad(u))), jump(grad(v), n)) * dS
        - inner(jump(grad(u), n), avg(div(grad(v)))) * dS
        + alpha / h_avg * inner(jump(grad(u), n), jump(grad(v), n)) * dS
    )

    # Define linear form
    L = f * v * dx

    # Solve variational problem
    u = Function(V)

    A, b = assemble_system(a, L, bc)
    A = A.array()
    b = b.get_local()
    x = np.linalg.solve(A, b)

    return A, b, x


def func(i, c1, c2, c3):
    A, b, x = generate_biharmonic_problem(c1, c2, c3)
    A = torch.from_numpy(A)
    b = torch.from_numpy(b)
    x = torch.from_numpy(x)
    torch.save([b, x], f"{i}_fu.pt")
    torch.save(A, f"{i}_A.pt")


# %%
@hydra.main(config_path="../configs", config_name="generate_biharmonic")
def main(cfg):
    c1s = loguniform.rvs(cfg.c1.min, cfg.c1.max, size=cfg.N_data)
    c2s = np.random.randint(cfg.c2.min, cfg.c2.max, size=cfg.N_data)
    c3s = np.random.randint(cfg.c3.min, cfg.c3.max, size=cfg.N_data)
    meta_df = pd.DataFrame({"c1": c1s, "c2": c2s, "c3": c3s})
    meta_df.to_csv("meta_df.csv", index=False)
    # func(0, 1, 1, 1)

    joblib.Parallel(n_jobs=-1)(
        joblib.delayed(func)(*params)
        for params in tqdm(meta_df.itertuples(), total=len(meta_df))
    )


if __name__ == "__main__":
    main()
