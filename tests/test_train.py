import pytest
import sys

from hydra.experimental import initialize, compose

from leela_zero_pytorch.train import main as train_main
from leela_zero_pytorch.weights import main as weights_main


@pytest.mark.parametrize("network_size", ["small", "big", "huge"])
def test_train_network_size(network_size):
    with initialize(config_path="../leela_zero_pytorch/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"+network={network_size}",
                "data.train_data_dir=tests/test-data",
                "data.train_dataloader_conf.batch_size=2",
                "data.val_data_dir=tests/test-data",
                "data.val_dataloader_conf.batch_size=2",
                "data.test_data_dir=tests/test-data",
                "data.test_dataloader_conf.batch_size=2",
                "+pl_trainer.fast_dev_run=true",
            ],
        )
        train_main(cfg)


def test_lzp_weights(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lzp-weights",
            "tests/test-data/epoch=0-step=0.ckpt",
            f"{tmp_path}/weights.txt",
        ],
    )
    weights_main()
