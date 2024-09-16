import pytest
from hydra import compose, initialize
from hydra.utils import instantiate
from nigbms.configs.data import OfflineDatasetConfig, OnlineDatasetConfig
from nigbms.tasks import PETScLinearSystemTask, PyTorchLinearSystemTask


class TestOfflineDataset:
    @pytest.fixture
    def init_dataset(self):
        with initialize(version_base="1.3"):
            cfg: OfflineDatasetConfig = compose(overrides=["+data@_global_=offline_dataset_default"])
            self.ds = instantiate(cfg)

    def test_load(self, init_dataset):
        task = self.ds.load(self.ds.data_dir / "0")
        assert isinstance(task, PETScLinearSystemTask)

    def test_getitem(self, init_dataset):
        tau = self.ds[0]
        assert isinstance(tau, PETScLinearSystemTask)


# TODO: parametrize this test with different task types
class TestOnlineDataset:
    @pytest.fixture
    def init_dataset(self, pytorch_task_constructor):
        with initialize(version_base="1.3"):
            cfg: OnlineDatasetConfig = compose(overrides=["+data@_global_=online_dataset_default"])
            self.ds = instantiate(cfg, constructor=pytorch_task_constructor)

    def test_iter(self, init_dataset):
        tau = next(iter(self.ds))
        assert isinstance(tau, PyTorchLinearSystemTask)
