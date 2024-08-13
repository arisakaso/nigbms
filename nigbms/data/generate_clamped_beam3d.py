from dataclasses import dataclass

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh
from mpi4py import MPI

import nigbms  # noqa
from nigbms.data.petsc import LinearProblem
from nigbms.modules.tasks import PETScLinearSystemTask, TaskParams


@dataclass
class ClampedBeam3DParams(TaskParams):
    L: float = 1.0
    W: float = 0.2
    mu: float = 1.0
    rho: float = 1.0
    lambda_: float = 1.25

    N_L: int = 20
    N_W: int = 6
    degree: int = 1
    rtol: float = 1e-6
    maxiter: int = 100


def construct_petsc_clamped_beam3d(params: ClampedBeam3DParams) -> PETScLinearSystemTask:
    # Reference: https://jsdokken.com/dolfinx-tutorial/chapter2/linearelasticity_code.html

    g = 0.4 * (params.W / params.L) ** 2

    domain = mesh.create_box(
        MPI.COMM_WORLD,
        [np.array([0, 0, 0]), np.array([params.L, params.W, params.W])],
        [params.N_L, params.N_W, params.N_W],
        cell_type=mesh.CellType.hexahedron,
    )
    V = fem.functionspace(domain, ("Lagrange", params.degree, (domain.geometry.dim,)))

    def clamped_boundary(x):
        return np.isclose(x[0], 0)

    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

    u_D = np.array([0, 0, 0], dtype=default_scalar_type)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
    T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
    ds = ufl.Measure("ds", domain=domain)

    def epsilon(u):
        return ufl.sym(ufl.grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)

    def sigma(u):
        return params.lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * params.mu * epsilon(u)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    f = fem.Constant(domain, default_scalar_type((0, 0, -params.rho * g)))
    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

    problem = LinearProblem(a, L, bcs=[bc])
    problem.assemble_system()

    # TODO: remove problem
    # If the problem object is not passed to the task, it will be garbage collected and A and b will be empty.
    # This is a workaround to keep the problem object alive.
    # A and b should be created without the problem object in the future.
    task = PETScLinearSystemTask(params, problem.A, problem.b, None, params.rtol, params.maxiter, problem)
    return task
