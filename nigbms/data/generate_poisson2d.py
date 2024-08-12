from dataclasses import dataclass

import hydra
import numpy as np
import ufl
from dolfinx.fem import (
    Function,
    dirichletbc,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from hydra.utils import instantiate
from mpi4py import MPI
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner

import nigbms  # noqa
from nigbms.data.petsc import LinearProblem
from nigbms.modules.tasks import PETScLinearSystemTask, TaskParams


@dataclass
class Poisson2DParams(TaskParams):
    coef: np.ndarray = np.array([1.0, 1.0])
    N: int = 10
    degree: int = 1
    rtol: float = 1e-6
    maxiter: int = 100


def construct_petsc_poisson2d_task(params: Poisson2DParams) -> PETScLinearSystemTask:
    # Reference: https://jsdokken.com/dolfinx-tutorial/chapter4/solvers.html

    def _u_ex(mod):
        return lambda x: mod.cos(params.coef[0] * mod.pi * x[0]) * mod.cos(params.coef[1] * mod.pi * x[1])

    u_numpy = _u_ex(np)
    u_ufl = _u_ex(ufl)
    mesh = create_unit_square(MPI.COMM_WORLD, params.N, params.N)
    x = SpatialCoordinate(mesh)
    f = -div(grad(u_ufl(x)))
    V = functionspace(mesh, ("Lagrange", params.degree))
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    L = f * v * dx
    u_bc = Function(V)
    u_bc.interpolate(u_numpy)
    facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True))
    dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
    bcs = [dirichletbc(u_bc, dofs)]
    problem = LinearProblem(a, L, bcs=bcs)
    problem.assemble_system()

    # TODO: remove problem
    # If the problem object is not passed to the task, it will be garbage collected and A and b will be empty.
    # This is a workaround to keep the problem object alive.
    # A and b should be created without the problem object in the future.
    task = PETScLinearSystemTask(params, problem.A, problem.b, None, params.rtol, params.maxiter, problem)
    return task


@hydra.main(version_base="1.3", config_path="../configs/data", config_name="generate_poisson2d")
def main(cfg):
    ds = instantiate(cfg.dataset)
    task = ds[1]
    task.b.view()


if __name__ == "__main__":
    main()
