import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from nigbms.configs.data import Poisson1DOfflineDataModuleConfig
from nigbms.modules.tasks import PETScLinearSystemTask, PyTorchLinearSystemTask


class TestOfflineDataset:
    @pytest.fixture
    def init_dataset(self):
        with initialize(version_base="1.3", config_path="../configs/modules/data"):
            cfg = compose(config_name="offline_dataset")
            self.ds = instantiate(cfg)

    def test_load(self, init_dataset):
        task = self.ds.load(self.ds.data_dir / "0")
        assert isinstance(task, PETScLinearSystemTask)

    def test_getitem(self, init_dataset):
        tau = self.ds[0]
        assert isinstance(tau, PETScLinearSystemTask)


class TestOnlineDataset:
    @pytest.fixture
    def init_dataset(self):
        with initialize(version_base="1.3", config_path="../configs/modules/data"):
            cfg = compose(config_name="online_dataset")
            self.ds = instantiate(cfg)

    def test_iter(self, init_dataset):
        tau = next(iter(self.ds))
        assert isinstance(tau, PETScLinearSystemTask)


class TestOfflineDataModule:
    @pytest.fixture
    def init_datamodule(self):
        with initialize(version_base="1.3", config_path="../configs/modules/data"):
            cfg = compose(config_name="offline_datamodule")
            self.dm = instantiate(cfg)

    def test_prepare_data(self, init_datamodule):
        self.dm.prepare_data()
        assert len(self.dm.indcs["test"]) == self.dm.dataset_sizes["test"]

    def test_setup(self, init_datamodule):
        self.dm.prepare_data()
        self.dm.setup()
        assert isinstance(self.dm.train_ds, torch.utils.data.Dataset)

    def test_train_dataloader(self, init_datamodule):
        self.dm.prepare_data()
        self.dm.setup()
        dl = self.dm.train_dataloader()
        batch = next(iter(dl))
        assert isinstance(batch[0], PETScLinearSystemTask)


class TestPoisson1DDataModule:
    @pytest.fixture
    def init_datamodule(self):
        with initialize(version_base="1.3"):
            cfg: Poisson1DOfflineDataModuleConfig = compose(overrides=["+data@_global_=poisson1d_offline_datamodule"])
            self.dm = instantiate(cfg)

    def test_prepare_data(self, init_datamodule):
        self.dm.prepare_data()
        assert len(self.dm.indcs["test"]) == self.dm.dataset_sizes["test"]

    def test_setup(self, init_datamodule):
        self.dm.prepare_data()
        self.dm.setup()
        assert isinstance(self.dm.train_ds, torch.utils.data.Dataset)

    def test_train_dataloader(self, init_datamodule):
        self.dm.prepare_data()
        self.dm.setup()
        dl = self.dm.train_dataloader()
        batch = next(iter(dl))
        assert isinstance(batch[0], PyTorchLinearSystemTask)
