from pathlib import Path

import hydra
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
from hydra.utils import instantiate
from mpi4py import MPI
from nigbms.tasks import (
    PETScLinearSystemTask,
    PyTorchLinearSystemTask,
    TaskConstructor,
    TaskParams,
    save_petsc_task,
    save_pytorch_task,
)
from nigbms.utils.petsc import LinearProblem
from tensordict import tensorclass
from torch import Tensor
from tqdm import tqdm
from ufl import SpatialCoordinate, TestFunction, TrialFunction, div, dx, grad, inner


# TODO: make this more complex
@tensorclass(autocast=True)
class Poisson2DParams(TaskParams):
    # physical parameters
    Tmax: Tensor = 100.0
    alpha: Tensor = 10.0
    u_center: Tensor = torch.tensor([0.5, 0.5])
    k_center: Tensor = torch.tensor([0.5, 0.5])

    # simulation parameters
    Nx: Tensor = 10
    Ny: Tensor = 10
    degree: Tensor = 1
    rtol: Tensor = 1e-6
    maxiter: Tensor = 100
    k_coef: Tensor = 0.0  # 0: isotropic, non 0: anisotropic


# TODO: make this faster
class Poisson2DTaskConstructor(TaskConstructor):
    def _u(self, mod, Tmax, alpha, center):
        return lambda x: Tmax * mod.exp(-alpha * ((x[0] - float(center[0])) ** 2 + (x[1] - float(center[1])) ** 2))

    def _k(self, k_coef, center):
        return lambda x: 1.0 + k_coef * ((x[0] - float(center[0])) ** 2 + (x[1] - float(center[1])) ** 2)

    def __call__(self, params: Poisson2DParams) -> PETScLinearSystemTask:
        # Reference: https://jsdokken.com/dolfinx-tutorial/chapter4/solvers.html

        # this is a workaround for the issue that ufl cannot handle torch.Tensor
        Tmax = params.Tmax.numpy()
        alpha = params.alpha.numpy()
        k_coef = params.k_coef.numpy()
        u_center = params.u_center.numpy()
        k_center = params.k_center.numpy()

        u_np = self._u(np, Tmax, alpha, u_center)
        u_ufl = self._u(ufl, Tmax, alpha, u_center)
        k_ufl = self._k(k_coef, k_center)

        # Define the variational problem
        mesh = create_unit_square(MPI.COMM_WORLD, params.Nx, params.Ny)
        x = SpatialCoordinate(mesh)
        f = -div(k_ufl(x) * grad(u_ufl(x)))
        V = functionspace(mesh, ("Lagrange", params.degree))
        u = TrialFunction(V)
        v = TestFunction(V)
        a = inner(k_ufl(x) * grad(u), grad(v)) * dx
        L = f * v * dx

        # Define the boundary condition
        u_bc = Function(V)
        u_bc.interpolate(u_np)
        facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, lambda x: np.full(x.shape[1], True))
        dofs = locate_dofs_topological(V, mesh.topology.dim - 1, facets)
        bcs = [dirichletbc(u_bc, dofs)]

        # Assemble the linear system
        problem = LinearProblem(a, L, bcs=bcs)
        problem.assemble_system()

        # TODO: remove problem
        # If the problem object is not passed to the task, it will be garbage collected and A and b will be empty.
        # This is a workaround to keep the problem object alive.
        # A and b should be created without the problem object in the future.
        task = PETScLinearSystemTask(params, problem.A, problem.b, None, params.rtol, params.maxiter, problem)

        return task


### Data generation script


@hydra.main(version_base="1.3", config_path=".", config_name="data_small")
def main(cfg) -> None:
    dataset = instantiate(cfg.dataset)
    for i in tqdm(range(cfg.N_data)):
        task = dataset[i]
        if isinstance(task, PyTorchLinearSystemTask):
            save_pytorch_task(task, Path(str(i)))
        elif isinstance(task, PETScLinearSystemTask):
            save_petsc_task(task, Path(str(i)))
        else:
            raise ValueError("Unknown task constructor.")


if __name__ == "__main__":
    main()