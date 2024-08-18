from hydra import compose, initialize

from nigbms.train.minimize_testfunctions import main


def test_main():
    with initialize(version_base="1.3", config_path="../configs/train"):
        cfg = compose(config_name="minimize_testfunctions")
        main(cfg)
