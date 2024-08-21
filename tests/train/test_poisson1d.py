from hydra import compose, initialize

from nigbms.train.poisson1d import main


# this test may fail when running all tests at once. please run again separately if it fails.
# TODO: why???
def test_main():
    with initialize(version_base="1.3", config_path="../configs/train"):
        cfg = compose(config_name="poisson1d")
        # workaround for hydra error: raise ValueError("HydraConfig was not set").
        # See https://github.com/facebookresearch/hydra/issues/2017
        cfg.wandb.project = "test_poisson1d"
        cfg.trainer.max_epochs = 1
        main(cfg)
