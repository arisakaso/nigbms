import numpy as np
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from nigbms.configs.data import Poisson1DOfflineDataModuleConfig
from nigbms.tasks import PETScLinearSystemTask, PyTorchLinearSystemTask
from numpy.testing import assert_array_equal
from sympy import pi, sin, symbols

from examples.poisson1d.task import (
    CoefsDistribution,
    Poisson1DParams,
    construct_pytorch_poisson1d_task,
    construct_sym_u,
    discretize,
    laplacian_matrix,
)


def test_laplacian_matrix() -> None:
    expected_result = np.array(
        [
            [2, -1, 0, 0, 0],
            [-1, 2, -1, 0, 0],
            [0, -1, 2, -1, 0],
            [0, 0, -1, 2, -1],
            [0, 0, 0, -1, 2],
        ]
    )
    assert_array_equal(laplacian_matrix(5), expected_result)


def test_discretize() -> None:
    x = symbols("x")
    expr = sin(x)
    N_grid = 128

    expected_result = np.sin(np.linspace(0, 1, N_grid))
    result = discretize(expr, N_grid)

    assert_array_equal(result, expected_result)


def test_construct_sym_u() -> None:
    N_terms = 3
    coefs = np.ones(3)
    u_sym = construct_sym_u(N_terms, coefs)
    assert u_sym == sum([coefs[i] * sin((i + 1) * pi * symbols("x")) for i in range(N_terms)])


def test_construct_pytorch_poisson1d_task() -> None:
    parasm = Poisson1DParams()
    task = construct_pytorch_poisson1d_task(parasm)
    assert isinstance(task, PyTorchLinearSystemTask)


class TestCoefsDistribution:
    def test_sample_multiple(self) -> None:
        dist = CoefsDistribution([10], 0.1, 1)
        coefs = dist.sample(0)
        assert isinstance(coefs, np.ndarray)
        assert coefs.shape == (10,)


@pytest.mark.parametrize("out_task_type", [PyTorchLinearSystemTask, PETScLinearSystemTask])
class TestPoisson1DDataModule:
    @pytest.fixture()
    def init_datamodule(self, out_task_type):
        with initialize(version_base="1.3"):
            cfg: Poisson1DOfflineDataModuleConfig = compose(overrides=["+data@_global_=poisson1d_offline_datamodule"])
            self.dm = instantiate(cfg)
            self.dm.out_task_type = out_task_type

    def test_prepare_data(self, init_datamodule):
        self.dm.prepare_data()
        assert len(self.dm.indcs["test"]) == self.dm.dataset_sizes["test"]

    def test_setup(self, init_datamodule, out_task_type):
        self.dm.prepare_data()
        self.dm.setup()
        assert isinstance(self.dm.train_ds, torch.utils.data.Dataset)

    def test_train_dataloader(self, init_datamodule, out_task_type):
        self.dm.prepare_data()
        self.dm.setup()
        dl = self.dm.train_dataloader()
        batch = next(iter(dl))
        assert isinstance(batch, out_task_type)
