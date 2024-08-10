# %%

# Reference: https://jsdokken.com/dolfinx-tutorial/chapter4/solvers.html

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
from nigbms.modules.data import PETScLinearSystemTask


def u_ex(mod, coefs):
    return lambda x: mod.cos(coefs[0] * mod.pi * x[0]) * mod.cos(coefs[1] * mod.pi * x[1])


def generate_petsc_poisson2d_problem(u_ex, N=10, degree=1) -> LinearProblem:
    u_numpy = u_ex(np)
    u_ufl = u_ex(ufl)
    mesh = create_unit_square(MPI.COMM_WORLD, N, N)
    x = SpatialCoordinate(mesh)
    f = -div(grad(u_ufl(x)))
    V = functionspace(mesh, ("Lagrange", degree))
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
    return problem


def generate_petsc_poisson2d_task(params) -> PETScLinearSystemTask:
    def _u_ex(mod):
        return lambda x: mod.cos(params[0] * mod.pi * x[0]) * mod.cos(params[1] * mod.pi * x[1])

    p = generate_petsc_poisson2d_problem(_u_ex)
    task = PETScLinearSystemTask(p.A, p.b, None, params.rtol, params.maxiter, params)
    return task
