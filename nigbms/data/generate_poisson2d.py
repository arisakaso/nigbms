# Reference: https://jsdokken.com/dolfinx-tutorial/chapter4/solvers.html

from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx.fem import (
    Function,
    dirichletbc,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from mpi4py import MPI
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner

from nigbms.data.petsc import LinearProblem
from nigbms.modules.tasks import PETScLinearSystemTask, TaskParams


@dataclass
class Poisson2DParams(TaskParams):
    coef1: float
    coef2: float
    N: int
    degree: int
    rtol: float
    maxiter: int


def generate_petsc_poisson2d_task(params: Poisson2DParams) -> PETScLinearSystemTask:
    def u_ex(mod):
        return lambda x: mod.cos(params.coef1 * mod.pi * x[0]) * mod.cos(params.coef2 * mod.pi * x[1])

    u_numpy = u_ex(np)
    u_ufl = u_ex(ufl)
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

    task = PETScLinearSystemTask(params, problem.A, problem.b, None, params.rtol, params.maxiter)
    return task


generate_petsc_poisson2d_task(Poisson2DParams(1, 1, 10, 1, 1e-6, 100))


# %%
