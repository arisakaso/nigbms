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
    coefs: Tensor = torch.ones(2)
    Nx: Tensor = 10
    Ny: Tensor = 10
    degree: Tensor = 1
    rtol: Tensor = 1e-6
    maxiter: Tensor = 100


# TODO: make this faster
class Poisson2DTaskConstructor(TaskConstructor):
    def _u_ex(self, mod, coefs):
        return lambda x: mod.cos(coefs[0] * mod.pi * x[0]) * mod.cos(coefs[1] * mod.pi * x[1])

    def __call__(self, params: Poisson2DParams) -> PETScLinearSystemTask:
        # Reference: https://jsdokken.com/dolfinx-tutorial/chapter4/solvers.html
        coefs = params.coefs.numpy()  # this is a workaround for the issue that ufl cannot handle torch.Tensor
        u_numpy = self._u_ex(np, coefs)
        u_ufl = self._u_ex(ufl, coefs)
        mesh = create_unit_square(MPI.COMM_WORLD, params.Nx, params.Ny)
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
