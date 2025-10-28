"""
Training script for Flow-based Diffusion Models
Does not modify main.py - standalone training script
"""

import argparse
import datetime
import os
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgm.util import instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser(description="Train Flow-based Diffusion Model")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., configs/training/flow_diffusion_mnist.yaml)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Experiment name",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="logs",
        help="Directory for logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="flow-diffusion",
        help="W&B project name",
    )

    return parser


def setup_logging(args, config):
    """Setup logging directory and logger"""
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    name = args.name if args.name else config.get("name", "flow_diffusion")
    logdir = Path(args.logdir) / f"{now}-{name}"
    logdir.mkdir(parents=True, exist_ok=True)

    ckptdir = logdir / "checkpoints"
    cfgdir = logdir / "configs"
    ckptdir.mkdir(exist_ok=True)
    cfgdir.mkdir(exist_ok=True)

    # Save config
    OmegaConf.save(config, cfgdir / f"{now}-project.yaml")

    # Setup logger
    if args.use_wandb:
        logger = WandbLogger(
            name=name,
            project=args.wandb_project,
            save_dir=str(logdir),
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(logdir),
            name="tensorboard",
        )

    return logdir, ckptdir, logger


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Set seed
    seed_everything(args.seed)

    # Load config
    config = OmegaConf.load(args.config)
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    # Setup logging
    logdir, ckptdir, logger = setup_logging(args, config)
    print(f"Logging to: {logdir}")

    # Instantiate model
    print("Instantiating model...")
    model_config = config.model
    model = instantiate_from_config(model_config)

    # Set learning rate
    if hasattr(model_config, "base_learning_rate"):
        model.learning_rate = model_config.base_learning_rate

    # Instantiate data
    print("Instantiating data loader...")
    data_config = config.data
    data = instantiate_from_config(data_config)

    # Setup callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckptdir,
        filename="{epoch:06}",
        verbose=True,
        save_last=True,
        every_n_train_steps=config.get("lightning", {}).get("modelcheckpoint", {}).get("params", {}).get("every_n_train_steps", 5000),
        save_top_k=config.get("lightning", {}).get("modelcheckpoint", {}).get("params", {}).get("save_top_k", 3),
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # Setup trainer
    trainer_config = config.get("lightning", {}).get("trainer", {})
    trainer_kwargs = dict(trainer_config)

    # Handle devices
    if "devices" in trainer_kwargs:
        devices = trainer_kwargs.pop("devices")
        if isinstance(devices, str):
            devices = [int(d) for d in devices.split(",") if d.strip()]
        trainer_kwargs["devices"] = devices if devices else 1

    trainer = pl.Trainer(
        default_root_dir=str(logdir),
        logger=logger,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    # Train
    print("Starting training...")
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.fit(model, data, ckpt_path=args.resume)
    else:
        trainer.fit(model, data)

    print("Training completed!")
    print(f"Checkpoints saved to: {ckptdir}")


if __name__ == "__main__":
    main()
