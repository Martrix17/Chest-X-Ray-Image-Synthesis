"""
Utility functions for saving and loading final model or checkpoint states.
"""

import json
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel, schedulers
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from models.label_conditioner import MultiLabelConditioner
from training.ema import EMA


def save_training_state(
    save_path: Path,
    step: int,
    optimizer: Optimizer,
    lr_scheduler: Optional[LRScheduler] = None,
) -> None:
    """
    Save training-only state (required for resuming).

    Args:
        save_path: Directory path to save to.
        step: Current step.
        optimizer: Optimizer instance.
        lr_scheduler: Learning rate scheduler instance.
    """
    states = {"step": step, "optimizer": optimizer.state_dict()}

    if lr_scheduler is not None:
        states.update({"lr_scheduler": lr_scheduler.state_dict()})

    torch.save(states, save_path / "training_state.pt")


def load_training_state(
    load_path: Path,
    optimizer: Optimizer,
    lr_scheduler: Optional[LRScheduler] = None,
    device: str = "cpu",
) -> int:
    """
    Load training state and last step from path.

    Args:
        load_path: Directory path to load from.
        optimizer: Optimizer instance.
        lr_scheduler: Learning rate scheduler instance.
        device: Device to load to.

    Returns:
        Last global step.
    """
    ckpt = torch.load(
        load_path / "training_state.pt",
        weights_only=False,
        map_location=device,
    )

    optimizer.load_state_dict(ckpt["optimizer"])

    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    return ckpt["step"]


def save_model(
    model: UNet2DConditionModel,
    conditioner: MultiLabelConditioner,
    scheduler: DDPMScheduler | DDIMScheduler,
    save_path: Path,
    accelerator: Accelerator,
    ema: EMA = None,
) -> None:
    """
    Save Unet model and components separately to local directory.

    Args:
        model: Unet model.
        conditioner: Label embedder.
        scheduler: Noise scheduler.
        save_path: Directory path to save to.
        accelerator: Accelerator instance.
        ema: EMA instance.
    """
    save_path.mkdir(parents=True, exist_ok=True)

    unet = accelerator.unwrap_model(model)
    conditioner = accelerator.unwrap_model(conditioner)

    torch.save(
        {
            "state_dict": unet.state_dict(),
            "config": unet.config,
        },
        save_path / "unet.pt",
    )

    torch.save(
        {
            "state_dict": conditioner.state_dict(),
            "config": {
                "num_classes": conditioner.num_classes,
                "embed_dim": conditioner.embed_dim,
            },
        },
        save_path / "conditioner.pt",
    )

    scheduler.save_config(save_path / "scheduler")

    if ema is not None:
        torch.save(
            {
                "state_dict": ema.state_dict(),
                "decay": ema.decay,
            },
            save_path / "ema.pt",
        )


def load_model_state(
    load_path: Path,
    model: UNet2DConditionModel,
    conditioner: MultiLabelConditioner,
    scheduler: DDPMScheduler | DDIMScheduler,
    device: str = "cpu",
) -> None:
    """
    Load Unet model and components from path.

    Args:
        load_path: Path to saved modules.
        model: Unet model.
        conditioner: Label embedder.
        scheduler: Noise scheduler.
        device: Device to load to (default: "cpu" for accelerator compatibility).
    """
    unet_ckpt = torch.load(load_path / "unet.pt", map_location=device, weights_only=False)
    model.load_state_dict(unet_ckpt["state_dict"])

    cond_ckpt = torch.load(load_path / "conditioner.pt", map_location=device, weights_only=False)
    conditioner.load_state_dict(cond_ckpt["state_dict"])

    scheduler_config_path = load_path / "scheduler" / "scheduler_config.json"
    with open(scheduler_config_path, "r") as f:
        config = json.load(f)

    scheduler_class_name = config.get("_class_name", "DDIMScheduler")
    SchedulerClass = getattr(schedulers, scheduler_class_name, None)
    if SchedulerClass is None:
        raise ValueError(f"Scheduler {scheduler_class_name} not found in diffusers.schedulers")

    scheduler_cfg = SchedulerClass.from_pretrained(load_path / "scheduler")
    scheduler.__dict__.update(scheduler_cfg.__dict__)


def load_ema_state(
    load_path: Path,
    ema: EMA,
    device: str = "cpu",
) -> None:
    """
    Load EMA instance from path.

    Args:
        load_path: Path to saved modules.
        ema: EMA instance.
        device: Device to load to (default: "cpu" for accelerator compatibility).
    """
    ema_path = load_path / "ema.pt"
    if not ema_path.exists():
        return

    ckpt = torch.load(ema_path, map_location=device, weights_only=False)
    ema.load_state_dict(ckpt["state_dict"])
