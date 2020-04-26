import hydra
import logging

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from leela_zero_pytorch.network import NetworkLightningModule
from leela_zero_pytorch.dataset import Dataset


logger = logging.getLogger(__name__)


@hydra.main(config_path="conf/config.yaml")
def main(cfg: DictConfig):
    logger.info(f"Training with the following config:\n{cfg.pretty()}")
    if cfg.train.resume_checkpoint:
        module = NetworkLightningModule.load_from_checkpoint(
            hydra.utils.to_absolute_path(cfg.train.resume_checkpoint)
        )
    else:
        module = NetworkLightningModule(
            {
                "board_size": cfg.network.board_size,
                "in_channels": cfg.network.in_channels,
                "residual_channels": cfg.network.residual_channels,
                "residual_layers": cfg.network.residual_layers,
                "learning_rate": cfg.train.learning_rate,
            }
        )
    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        gpus=cfg.train.gpus,
        early_stop_callback=cfg.train.early_stop,
        distributed_backend="ddp"
        if cfg.train.gpus is not None and cfg.train.gpus > 1
        else None,
        train_percent_check=cfg.train.train_percent,
        val_percent_check=cfg.train.val_percent,
    )
    trainer.fit(
        module,
        train_dataloader=DataLoader(
            Dataset.from_data_dir(
                hydra.utils.to_absolute_path(cfg.dataset.train_dir), transform=True
            ),
            shuffle=True,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.n_data_workers,
        ),
        val_dataloaders=DataLoader(
            Dataset.from_data_dir(hydra.utils.to_absolute_path(cfg.dataset.val_dir)),
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.n_data_workers,
        ),
        test_dataloaders=DataLoader(
            Dataset.from_data_dir(hydra.utils.to_absolute_path(cfg.dataset.test_dir)),
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.n_data_workers,
        ),
    )
    if cfg.train.run_test:
        trainer.test()


# this function is required to allow automatic detection of the module name when running
# from a binary script.
# it should be called from the executable script and not the hydra.main() function directly.
def entry():
    main()


if __name__ == "__main__":
    main()
