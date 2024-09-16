import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from nigbms.configs.data import Poisson1DOfflineDataModuleConfig
from nigbms.tasks import (
    PETScLinearSystemTask,
    PyTorchLinearSystemTask,
)

from examples.poisson2d.task import Poisson2DParams, Poisson2DTaskConstructor


class TestPoisson2DTaskConstructor:
    def setup_method(self):
        self.constructor = Poisson2DTaskConstructor()

    def test_construct_task(self) -> None:
        params = Poisson2DParams()
        task = self.constructor(params)
        assert isinstance(task, PETScLinearSystemTask)


@pytest.mark.parametrize("out_task_type", [PyTorchLinearSystemTask, PETScLinearSystemTask])
class TestPoisson2DDataModule:
    @pytest.fixture
    def init_datamodule(self, out_task_type):
        with initialize(version_base="1.3"):
            cfg: Poisson1DOfflineDataModuleConfig = compose(overrides=["+data@_global_=poisson2d_offline_datamodule"])
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
