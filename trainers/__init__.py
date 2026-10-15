"""
Training scripts for iRea models.

This module provides trainers for various recommendation models:

Trainers:
    - general_trainer: Shared SID NTP training for TIGER, SASRec and HSTU
    - rqvae_trainer: RQVAE training with collision rate metrics

Utilities:
    - trainer_utils: Common functions (setup_accelerator, setup_wandb, etc.)

Usage:
    python -m trainers.general_trainer config/<model>.gin
"""

from trainers.trainer_utils import (
    TrainerConfig,
    setup_accelerator,
    setup_wandb,
    save_checkpoint,
    load_checkpoint,
    get_parameter_count,
    log_training_info,
)

__all__ = [
    "general_trainer",
    "rqvae_trainer",
    # Utilities
    "TrainerConfig",
    "setup_accelerator",
    "setup_wandb",
    "save_checkpoint",
    "load_checkpoint",
    "get_parameter_count",
    "log_training_info",
]
