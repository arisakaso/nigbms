from hydra import compose, initialize

from nigbms.train.minimize_testfunctions import main


def test_main():
    with initialize(version_base="1.3", config_path="../configs/train"):
        cfg = compose(config_name="minimize_testfunctions")
        y_mean = main(cfg)
        assert y_mean < 1.0e-6
