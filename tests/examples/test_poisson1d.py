import numpy as np
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from nigbms.configs.data import Poisson1DOfflineDataModuleConfig
from nigbms.tasks import PETScLinearSystemTask, PyTorchLinearSystemTask
from numpy.testing import assert_array_equal

from examples.poisson1d.task import (
    CoefsDistribution,
    Poisson1DParams,
    Poisson1DTaskConstructor,
)


class TestPoisson1DTaskConstructor:
    def setup_method(self):
        self.constructor = Poisson1DTaskConstructor()

    def test_laplacian_matrix(self) -> None:
        expected_result = np.array(
            [
                [2, -1, 0, 0, 0],
                [-1, 2, -1, 0, 0],
                [0, -1, 2, -1, 0],
                [0, 0, -1, 2, -1],
                [0, 0, 0, -1, 2],
            ]
        )
        assert_array_equal(self.constructor._laplacian_matrix(5), expected_result)

    def test_discretize(self) -> None:
        N_grid = 128
        expected_result = np.sin(np.pi * np.linspace(0, 1, N_grid))
        result = self.constructor._discretize(1, np.array([1.0]), N_grid)
        assert_array_equal(result, expected_result)

    def test_construct_task(self) -> None:
        params = Poisson1DParams()
        task = self.constructor(params)
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


# TODO: Implement test for train.py
def test_train():
    pass
