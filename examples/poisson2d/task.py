# %%
import nigbms  # noqa
import numpy as np
import torch
import ufl
from dolfinx.fem import (
    Function,
    dirichletbc,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from mpi4py import MPI
from nigbms.tasks import PETScLinearSystemTask, TaskParams
from nigbms.utils.petsc import LinearProblem
from tensordict import tensorclass
from torch import Tensor
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner


@tensorclass(autocast=True)
class Poisson2DParams(TaskParams):
    coef: Tensor = torch.ones(2)
    N: Tensor = 10
    degree: Tensor = 1
    rtol: Tensor = 1e-6
    maxiter: Tensor = 100


def construct_petsc_poisson2d_task(params: Poisson2DParams) -> PETScLinearSystemTask:
    # Reference: https://jsdokken.com/dolfinx-tutorial/chapter4/solvers.html
    coef = params.coef.numpy()  # this is a workaround for the issue that ufl cannot handle torch.Tensor

    def _u_ex(mod):
        return lambda x: mod.cos(coef[0] * mod.pi * x[0]) * mod.cos(coef[1] * mod.pi * x[1])

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
