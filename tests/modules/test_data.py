import pandas as pd
import pytest
import torch
from hydra import compose, initialize
from hydra.utils import instantiate

from nigbms.modules.tasks import PETScLinearSystemTask, PyTorchLinearSystemTask


class TestOfflineDataset:
    @pytest.fixture
    def init_dataset(self):
        with initialize(version_base="1.3", config_path="../configs/modules"):
            cfg = compose(config_name="data")
            self.ds = instantiate(cfg.offline_dataset)
            self.ds.meta_df = pd.read_csv(self.ds.data_dir + "/meta_df.csv")

    def test_load(self, init_dataset):
        A = self.ds.load(self.ds.data_dir + "/A")
        assert isinstance(A, torch.Tensor)

    def test_getitem(self, init_dataset):
        tau = self.ds[0]
        assert isinstance(tau, PyTorchLinearSystemTask)


class TestOnlineDataset:
    @pytest.fixture
    def init_dataset(self):
        with initialize(version_base="1.3", config_path="../configs/modules"):
            cfg = compose(config_name="data")
            self.ds = instantiate(cfg.online_dataset)

    def test_iter(self, init_dataset):
        tau = next(iter(self.ds))
        assert isinstance(tau, PETScLinearSystemTask)
