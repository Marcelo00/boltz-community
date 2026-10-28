import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import listconfig
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from boltz.data.module.trainingv3 import BoltzTrainingDataModule, DataConfig
from boltz.model.models.boltz2 import Boltz2


@dataclass
class TrainConfig:
    """Train configuration.

    Attributes
    ----------
    data : DataConfig
        The data configuration.
    pretrained : str
        The Boltz-2 checkpoint to fine-tune.
    model : dict
        The hyperparameters overriding the ones stored in the checkpoint.
    output : str
        The output directory.
    trainer : Optional[dict]
        The trainer configuration.
    resume : Optional[str]
        The resume checkpoint.
    wandb : Optional[dict]
        The wandb configuration.
    matmul_precision : Optional[str]
        The matmul precision.
    find_unused_parameters : Optional[bool]
        Find unused parameters.
    save_top_k : Optional[int]
        Save the top k checkpoints by validation lddt.
    validation_only : bool
        Run validation only, on the pretrained or the resumed model.
    debug : bool
        Debug mode.

    """

    data: DataConfig
    pretrained: str
    model: dict
    output: str
    trainer: Optional[dict] = None
    resume: Optional[str] = None
    wandb: Optional[dict] = None
    matmul_precision: Optional[str] = None
    find_unused_parameters: Optional[bool] = False
    save_top_k: Optional[int] = 1
    validation_only: bool = False
    debug: bool = False


def load_model(checkpoint: str, model_args: dict) -> Boltz2:
    """Load a pretrained Boltz-2, keeping the checkpoint architecture.

    Parameters
    ----------
    checkpoint : str
        The path to the checkpoint.
    model_args : dict
        The hyperparameters overriding the ones in the checkpoint.

    Returns
    -------
    Boltz2
        The loaded model.

    """
    # Lightning checkpoints hold OmegaConf objects, which torch's safe loading rejects
    return Boltz2.load_from_checkpoint(
        checkpoint,
        map_location="cpu",
        strict=False,
        weights_only=False,
        **model_args,
    )


@rank_zero_only
def save_config_to_wandb(logger: WandbLogger, raw_config: omegaconf.DictConfig) -> None:
    """Upload the run configuration to wandb."""
    config_path = Path(logger.experiment.dir) / "run.yaml"
    omegaconf.OmegaConf.save(raw_config, config_path)
    logger.experiment.save(str(config_path))


def train(raw_config: str, args: list[str]) -> None:
    """Run training.

    Parameters
    ----------
    raw_config : str
        The input yaml configuration.
    args : list[str]
        Any command line overrides.

    """
    raw_config = omegaconf.OmegaConf.load(raw_config)
    args = omegaconf.OmegaConf.from_dotlist(args)
    raw_config = omegaconf.OmegaConf.merge(raw_config, args)

    cfg = hydra.utils.instantiate(raw_config)
    cfg = TrainConfig(**cfg)

    if cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    trainer = cfg.trainer if cfg.trainer is not None else {}
    devices = trainer.get("devices", 1)

    wandb = cfg.wandb
    if cfg.debug:
        if isinstance(devices, int):
            devices = 1
        elif isinstance(devices, (list, listconfig.ListConfig)):
            devices = [devices[0]]
        trainer["devices"] = devices
        cfg.data.num_workers = 0
        wandb = None

    data_module = BoltzTrainingDataModule(DataConfig(**cfg.data))
    model_module = load_model(cfg.pretrained, cfg.model)

    callbacks = [
        ModelCheckpoint(
            monitor="val/lddt",
            mode="max",
            save_top_k=cfg.save_top_k,
            save_last=True,
        )
    ]

    loggers = []
    if wandb:
        wandb_logger = WandbLogger(
            name=wandb["name"],
            group=wandb["name"],
            save_dir=cfg.output,
            project=wandb["project"],
            entity=wandb["entity"],
            log_model=False,
        )
        save_config_to_wandb(wandb_logger, raw_config)
        loggers.append(wandb_logger)

    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, (list, listconfig.ListConfig)) and len(devices) > 1
    ):
        strategy = DDPStrategy(find_unused_parameters=cfg.find_unused_parameters)

    trainer = pl.Trainer(
        default_root_dir=str(cfg.output),
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        reload_dataloaders_every_n_epochs=1,
        **trainer,
    )
    run = trainer.validate if cfg.validation_only else trainer.fit
    run(model_module, datamodule=data_module, ckpt_path=cfg.resume)


if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2:])
