from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from nigbms.tasks import (
    PETScLinearSystemTask,
    PyTorchLinearSystemTask,
    TaskConstructor,
    TaskParams,
    save_petsc_task,
    save_pytorch_task,
)
from nigbms.utils.distributions import Distribution
from tensordict import tensorclass
from torch import Tensor, tensor
from tqdm import tqdm


@tensorclass(autocast=True)
class Poisson1DParams(TaskParams):
    N_terms: Tensor = 10
    N_grid: Tensor = 100
    coefs: Tensor = torch.ones(10)
    rtol: Tensor = 1e-6
    maxiter: Tensor = 100


class CoefsDistribution(Distribution):
    def __init__(self, shape, p: float, scale: float):
        assert len(shape) == 1
        super().__init__(shape)
        self.p = p
        self.scale = scale

    def sample(self, seed: int = None) -> np.ndarray:
        np.random.seed(seed)
        is_difficult = np.random.choice([True, False], p=[self.p, 1 - self.p])
        if is_difficult:
            return np.random.normal(0, self.scale * np.linspace(-1, 1, self.shape[0]) ** 2, self.shape[0])
        else:
            return np.random.normal(0, self.scale * (1 - np.abs(np.linspace(-1, 1, self.shape[0]))), self.shape[0])


class Poisson1DTaskConstructor(TaskConstructor):
    def _laplacian_matrix(self, N: int) -> np.ndarray:
        """
        Generate a Laplacian matrix of size N.

        Parameters:
        - N (int): The size of the Laplacian matrix.

        Returns:
        - numpy.ndarray: The Laplacian matrix.
        """
        D = np.zeros((N, N))
        D += np.diag([-1] * (N - 1), k=-1)
        D += np.diag([2] * N, k=0)
        D += np.diag([-1] * (N - 1), k=1)
        return D

    def _discretize(self, N_terms: int, coefs: np.ndarray, N_grid: int) -> np.ndarray:
        """
        Discretizes the given coefficients in [0, 1] over a specified number of grid points.
        This includes the values at the boundaries.

        Parameters:
        N_terms (int): The number of terms in the series.
        coefs (np.ndarray): The coefficients of the series.
        N_grid (int): The number of grid points.

        Returns:
        numpy.ndarray: An array containing the discretized values.
        """
        x = np.linspace(0, 1, N_grid)
        u = sum([coefs[i] * np.sin((i + 1) * np.pi * x) for i in range(N_terms)])
        return u

    def __call__(self, params: Poisson1DParams) -> PyTorchLinearSystemTask:
        x = self._discretize(params.N_terms, params.coefs.numpy(), params.N_grid + 2)[1:-1]
        x = x.reshape(params.N_grid, 1)  # column vector
        A = self._laplacian_matrix(params.N_grid)
        b = A @ x
        task = PyTorchLinearSystemTask(
            params,
            tensor(A),
            tensor(b),
            tensor(x),
            params.rtol,
            params.maxiter,
        )
        return task


### Data generation script


@hydra.main(version_base="1.3", config_path=".", config_name="data_small")
def main(cfg) -> None:
    dataset = instantiate(cfg.dataset)
    import os

    print(os.getcwd())
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
