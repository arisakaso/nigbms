from hydra import compose, initialize

from nigbms.train.poisson1d import main


def test_main():
    with initialize(version_base="1.3", config_path="../configs/train"):
        cfg = compose(config_name="poisson1d")
        cfg.wandb.project = "test"  # workaround for hydra error: raise ValueError("HydraConfig was not set"). see https://github.com/facebookresearch/hydra/issues/2017
        cfg.trainer.max_epochs = 1
        main(cfg)
