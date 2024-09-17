import nigbms  # noqa
import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh
from mpi4py import MPI
from nigbms.tasks import PETScLinearSystemTask, TaskParams
from nigbms.utils.petsc import LinearProblem
from tensordict import tensorclass
from torch import Tensor


@tensorclass
class Beam3DParams(TaskParams):
    L: Tensor = 1.0
    W: Tensor = 0.2
    mu: Tensor = 1.0
    rho: Tensor = 1.0
    lambda_: Tensor = 1.25

    N_L: Tensor = 20
    N_W: Tensor = 6
    degree: Tensor = 1
    rtol: Tensor = 1e-6
    maxiter: Tensor = 100


class Beam3DTaskConstructor:
    def _clamped_boundary(self, x):
        return np.isclose(x[0], 0)

    def _epsilon(self, u):
        return ufl.sym(ufl.grad(u))

    def _sigma(self, u, lambda_, mu):
        return lambda_ * ufl.div(u) * ufl.Identity(len(u)) + 2 * mu * self._epsilon(u)

    def __call__(self, params: Beam3DParams) -> PETScLinearSystemTask:
        g = 0.4 * (params.W / params.L) ** 2

        domain = mesh.create_box(
            MPI.COMM_WORLD,
            [np.array([0, 0, 0]), np.array([params.L, params.W, params.W])],
            [params.N_L, params.N_W, params.N_W],
            cell_type=mesh.CellType.hexahedron,
        )
        V = fem.functionspace(domain, ("Lagrange", params.degree, (domain.geometry.dim,)))

        fdim = domain.topology.dim - 1
        boundary_facets = mesh.locate_entities_boundary(domain, fdim, self._clamped_boundary)

        u_D = np.array([0, 0, 0], dtype=default_scalar_type)
        bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
        T = fem.Constant(domain, default_scalar_type((0, 0, 0)))
        ds = ufl.Measure("ds", domain=domain)

        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        f = fem.Constant(domain, default_scalar_type((0, 0, -params.rho * g)))
        a = ufl.inner(self._sigma(u, params.lambda_, params.mu), self._epsilon(v)) * ufl.dx
        L = ufl.dot(f, v) * ufl.dx + ufl.dot(T, v) * ds

        problem = LinearProblem(a, L, bcs=[bc])
        problem.assemble_system()

        # TODO: remove problem
        # If the problem object is not passed to the task, it will be garbage collected and A and b will be empty.
        # This is a workaround to keep the problem object alive.
        # A and b should be created without the problem object in the future.
        task = PETScLinearSystemTask(params, problem.A, problem.b, None, params.rtol, params.maxiter, problem)
        return task
